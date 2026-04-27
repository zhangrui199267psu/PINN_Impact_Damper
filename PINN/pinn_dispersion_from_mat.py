"""
pinn_dispersion_from_mat.py
---------------------------
Approach 2 — Finite-lattice PINN-based dispersion (Stage 2 only).

Two-stage workflow
~~~~~~~~~~~~~~~~~~
Stage 1  (PINN_ndofs_nmasses/pinn_ndof_chain_sim.ipynb):
    Run the segment-by-segment PINN solver → stitch the spatiotemporal
    field  x_n(t).  Save to a .mat file via  save_pinn_results().

Stage 2  (dispersion_ndofs/pinn_dispersion_from_mat.ipynb, THIS MODULE):
    Load the .mat file.  No PINN retraining required.
    Resample x_n(t) to a uniform time grid.
    Apply 2-D FFT → S(k, ω).
    Plot dispersion heatmap, DOS profile, and per-band ridge curves.

What makes this the "real" dispersion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  • Finite lattice with actual boundary conditions (chain or ring as set up)
  • Full nonsmooth dynamics: impact times found exactly by root-finding
  • No Bloch periodicity assumption — each cell can behave differently
  • Amplitude / forcing dependence is captured automatically
  • Multi-band structure (harmonics, band gaps) emerges directly from the data

Key difference from Bloch (Approach 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Bloch assumes an infinite periodic lattice; this approach uses the real
  finite simulation data, including boundary effects and spatial non-uniformity
  that develop in strongly nonlinear or spatially varying regimes.

Author: Rui Zhang
"""

import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from Dispersion.dispersion_theory import impact_damper_branches, mass_in_mass_dispersion


# ---------------------------------------------------------------------------
# Save / load helpers (used by pinn_ndof_chain_sim.ipynb and this notebook)
# ---------------------------------------------------------------------------

def save_pinn_results(filepath, t_total, x_PINN_total, params=None):
    """
    Save stitched PINN simulation results to a .mat file.

    Call this at the end of pinn_ndof_chain_sim.ipynb.

    Parameters
    ----------
    filepath       : str  — output .mat path, e.g. 'pinn_results_ndof10.mat'
    t_total        : (N_time,) global time array  (may be non-uniform)
    x_PINN_total   : (N_time, n_dof)  PINN primary-mass displacements
    params         : dict of scalar/array parameters to embed, e.g.
                         {'n_dof': 10, 'mx': 1.0, 'my': 0.1,
                          'k': 100.0, 'c': 0.5, 'D': 1.0, 'r': 0.7,
                          'phi1': 1.0, 'phi2': 1.0}
    """
    out = {
        't_total':      t_total.flatten(),
        'x_PINN_total': x_PINN_total,       # (N_time, n_dof)
    }
    if params is not None:
        for key, val in params.items():
            out[key] = np.atleast_1d(np.asarray(val, dtype=float))

    sio.savemat(filepath, out)
    print(f'Saved → {filepath}')
    print(f'  t_total      : {t_total.shape}')
    print(f'  x_PINN_total : {x_PINN_total.shape}')


def load_pinn_results(filepath):
    """
    Load PINN simulation results from a .mat file.

    Returns
    -------
    t_total      : (N_time,) time array
    x_PINN_total : (N_time, n_dof)  primary displacements
    params       : dict  — all saved scalar / array parameters
    """
    data = sio.loadmat(filepath)

    t_total      = data['t_total'].flatten()
    x_PINN_total = np.asarray(data['x_PINN_total'])

    # Ensure shape (N_time, n_dof)
    if x_PINN_total.ndim == 2 and x_PINN_total.shape[0] < x_PINN_total.shape[1]:
        x_PINN_total = x_PINN_total.T

    skip = {'__header__', '__version__', '__globals__', 't_total', 'x_PINN_total'}
    params = {}
    for key, val in data.items():
        if key in skip:
            continue
        arr = np.asarray(val).flatten()
        params[key] = float(arr[0]) if arr.size == 1 else arr

    print(f'Loaded  {filepath}')
    print(f'  t_total      : {t_total.shape},  span {t_total[0]:.3f} – {t_total[-1]:.3f} s')
    print(f'  x_PINN_total : {x_PINN_total.shape}')
    if params:
        print(f'  params       : {list(params.keys())}')

    return t_total, x_PINN_total, params


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_uniform(t_raw, X_raw, omega_max_lin, n_harmonic=4, pts_per_cycle=10):
    """
    Resample the (possibly non-uniform) spatiotemporal array to a uniform grid.

    The target dt is chosen to resolve up to  n_harmonic × ω_max_lin
    with at least pts_per_cycle samples per shortest cycle.

    Parameters
    ----------
    t_raw         : (M,) time array  (may have variable spacing at segment joints)
    X_raw         : (n_dof, M)  spatiotemporal displacements
    omega_max_lin : float  — top of the linear acoustic branch  2√(K/M)
    n_harmonic    : int    — number of harmonic bands to resolve (default 4)
    pts_per_cycle : int    — minimum samples per cycle of highest frequency

    Returns
    -------
    t_uniform     : (P,) uniform time array
    X_uniform     : (n_dof, P)  resampled array
    dt            : float  — uniform time step
    omega_nyquist : float  — Nyquist frequency of the new grid
    """
    omega_max_resolve = n_harmonic * omega_max_lin
    dt = (2.0 * np.pi / omega_max_resolve) / pts_per_cycle

    t_uniform = np.arange(t_raw[0], t_raw[-1], dt)
    n_dof     = X_raw.shape[0]
    X_uniform = np.zeros((n_dof, len(t_uniform)))

    for i in range(n_dof):
        f = interp1d(t_raw, X_raw[i, :], kind='linear', fill_value='extrapolate')
        X_uniform[i, :] = f(t_uniform)

    omega_nyquist = np.pi / dt
    return t_uniform, X_uniform, dt, omega_nyquist


# ---------------------------------------------------------------------------
# Spectral analysis helpers
# ---------------------------------------------------------------------------

def dispersion_from_2dfft(t, x_nt, skip_transient=0.25):
    """
    Extract the frequency–wavenumber spectrum via 2-D FFT.

    Parameters
    ----------
    t             : (T,) uniform time array
    x_nt          : (N, T) spatiotemporal displacement  (N = n_dof, T = time steps)
    skip_transient: fraction of T to discard at the start (removes startup transient)

    Returns
    -------
    k_pos    : (N//2+1,) positive wavenumbers  [0, π]  rad/unit-cell
    omega_pos: (T//2+1,) positive angular frequencies  rad/s
    spectrum : (N//2+1, T//2+1)  one-sided power spectrum  |FFT_2D|²
    """
    N, T = x_nt.shape
    dt   = float(t[1] - t[0])

    i0   = int(skip_transient * T)
    data = x_nt[:, i0:]
    T2   = data.shape[1]

    F    = np.fft.fft2(data)

    k_all = 2.0 * np.pi * np.fft.fftfreq(N)
    k_pos = np.abs(k_all[:N // 2 + 1])

    om_all = 2.0 * np.pi * np.fft.fftfreq(T2, d=dt)
    om_pos = om_all[:T2 // 2 + 1]

    spectrum = np.abs(F) ** 2
    spectrum = spectrum[:N // 2 + 1, :T2 // 2 + 1]

    return k_pos, om_pos, spectrum


def linear_dispersion(k_wavenumber, k_coupling, mx):
    """
    Acoustic branch of a monoatomic ring chain (no impactor, no damping):

        ω_lin(k) = 2 · √(K/M) · |sin(k/2)|
    """
    k = np.asarray(k_wavenumber, dtype=float)
    return 2.0 * np.sqrt(k_coupling / mx) * np.abs(np.sin(k / 2.0))


def extract_ridge(k_pos, omega_pos, spectrum, omega_min=0.01):
    """
    Extract the dominant ω for each k from the 2-D spectrum (full-range ridge).

    Parameters
    ----------
    k_pos, omega_pos, spectrum : from dispersion_from_2dfft()
    omega_min : ignore frequencies below this (avoids DC peak)

    Returns
    -------
    k_pos       : same as input
    omega_ridge : (N//2+1,) dominant ω at each k  (NaN if no peak)
    """
    i_min = np.searchsorted(omega_pos, omega_min)
    omega_ridge = np.full(len(k_pos), np.nan)
    for i in range(len(k_pos)):
        row = spectrum[i, i_min:]
        if row.max() > 0:
            omega_ridge[i] = omega_pos[i_min + int(np.argmax(row))]
    return k_pos, omega_ridge


def plot_dispersion_heatmap(
    k_pos, omega_pos, spectrum,
    k_coupling, mx,
    dB_range=40.0,
    save_path=None,
    figsize=(7, 5),
    omega_max=None,
    impact_curve_params=None,
):
    """
    Plot |FFT_2D(x_n(t))|² as a (k, ω) heatmap with the linear dispersion overlay.

    Parameters
    ----------
    k_pos, omega_pos, spectrum : from dispersion_from_2dfft()
    k_coupling, mx : lattice parameters
    dB_range       : dynamic range to display in dB (default 40 dB)
    save_path      : optional file path
    figsize        : (w, h) inches
    omega_max      : clip ω axis (None = 1.5 × linear max)
    impact_curve_params : dict or None
        Optional overlay for impact-damper branches. Example:
            {
                'p': 1.0,
                'm': 1.0,
                'k_coupling': 1.0,
                'mu': 0.3,
                'p_ref': 1.0,
                'n_subharmonic': 8,
            }

    Returns
    -------
    fig, ax
    """
    mpl.rcParams.update({
        'font.family':      'Times New Roman',
        'mathtext.fontset': 'custom',
        'mathtext.rm':      'Times New Roman',
        'mathtext.it':      'Times New Roman:italic',
        'mathtext.bf':      'Times New Roman:bold',
        'pdf.fonttype':     42,
        'ps.fonttype':      42,
    })
    FS = 20
    LW = 2.0

    if omega_max is None:
        omega_max = 1.5 * linear_dispersion(np.pi, k_coupling, mx)
    i_om   = np.searchsorted(omega_pos, omega_max)
    om_plt = omega_pos[:i_om]
    sp_plt = spectrum[:, :i_om]

    S_dB = 10.0 * np.log10(sp_plt + 1e-30)
    S_dB -= S_dB.max()
    S_dB  = np.clip(S_dB, -dB_range, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(k_pos / np.pi, om_plt, S_dB.T,
                       cmap='inferno', shading='auto',
                       vmin=-dB_range, vmax=0.0)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Spectral power (dB)', fontsize=FS - 4)
    cbar.ax.tick_params(labelsize=FS - 6)

    k_line = np.linspace(0, np.pi, 300)
    ax.plot(k_line / np.pi, linear_dispersion(k_line, k_coupling, mx),
            'w--', lw=LW, label=r'Linear  ($D \to \infty$)')

    if impact_curve_params is not None:
        pars = {
            'p': impact_curve_params.get('p', 1.0),
            'm': impact_curve_params.get('m', mx),
            'k_coupling': impact_curve_params.get('k_coupling', k_coupling),
            'mu': impact_curve_params.get('mu', 0.3),
            'p_ref': impact_curve_params.get('p_ref', 1.0),
            'n_subharmonic': impact_curve_params.get('n_subharmonic', 8),
        }
        impact = impact_damper_branches(k_line, **pars)
        ax.plot(k_line / np.pi, impact['omega_acoustic'],
                color='cyan', lw=1.8, ls='-', label='Impact acoustic')
        ax.plot(k_line / np.pi, impact['omega_optical'],
                color='cyan', lw=1.8, ls='-.', label='Impact optical')
        for i, om_sh in enumerate(impact['omega_subharmonics']):
            ax.plot(k_line / np.pi, om_sh, color='cyan', lw=0.8, alpha=0.28,
                    label='Impact subharmonics' if i == 0 else None)

    ax.set_xlabel(r'Wavenumber  $k / \pi$',        fontsize=FS, labelpad=8)
    ax.set_ylabel(r'Frequency  $\omega$  (rad/s)',  fontsize=FS, labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, omega_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    ax.tick_params(axis='both', labelsize=FS - 2)
    ax.legend(fontsize=FS - 5, loc='upper left', framealpha=0.6)
    plt.tight_layout(pad=1.5)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()
    return fig, ax


def compute_spectrum(t_uniform, X_uniform, skip_transient=0.15):
    """
    Compute the 2-D power spectrum S(k, ω) via 2-D FFT.

    Parameters
    ----------
    t_uniform     : (P,) uniform time array
    X_uniform     : (n_dof, P)
    skip_transient: fraction of P to discard before FFT

    Returns
    -------
    k_pos    : (n_dof//2+1,) positive wavenumbers  [0, π]  rad/unit-cell
    omega_pos: (P//2+1,) positive angular frequencies  rad/s
    spectrum : (n_dof//2+1, P//2+1)  one-sided power spectrum
    """
    return dispersion_from_2dfft(t_uniform, X_uniform, skip_transient=skip_transient)


def clip_spectrum(k_pos, omega_pos, spectrum, omega_max):
    """Clip spectrum to [0, omega_max] on the ω axis."""
    i_max    = np.searchsorted(omega_pos, omega_max)
    return k_pos, omega_pos[:i_max], spectrum[:, :i_max]


def compute_dos(spectrum):
    """
    Density of States: integrate S(k, ω) over all k.

    Returns
    -------
    dos    : (n_omega,) k-integrated power spectrum
    dos_dB : normalised dB version (0 dB = maximum)
    """
    dos    = spectrum.sum(axis=0)
    dos_dB = 10.0 * np.log10(dos + 1e-30)
    dos_dB -= dos_dB.max()
    return dos, dos_dB


def detect_band_gaps(omega_plot, dos_dB, dip_threshold_dB=5.0, min_distance=5):
    """
    Identify band-gap centre frequencies from the DOS curve.

    A band gap is a significant dip (valley) in the DOS.

    Parameters
    ----------
    omega_plot        : (n_omega,) frequency array
    dos_dB            : (n_omega,) normalised DOS in dB
    dip_threshold_dB  : minimum depth of a dip to be called a gap (default 5 dB)
    min_distance      : minimum spacing between consecutive gaps (in index units)

    Returns
    -------
    gap_freqs  : array of gap-centre frequencies  (rad/s)
    band_freqs : array of pass-band peak frequencies  (rad/s)
    """
    neg_dos = -dos_dB
    gap_idx,  _ = find_peaks(neg_dos,  height=dip_threshold_dB, distance=min_distance)
    band_idx, _ = find_peaks(dos_dB,   height=-30.0,            distance=min_distance)
    return omega_plot[gap_idx], omega_plot[band_idx]


def extract_ridge_in_band(k_pos, omega_band, spec_band, omega_min=0.0):
    """
    Extract dominant ω at each k within a frequency sub-band.

    Parameters
    ----------
    k_pos      : (N_k,) wavenumber array
    omega_band : (N_ω,) frequency array for this band
    spec_band  : (N_k, N_ω) spectrum slice
    omega_min  : skip frequencies below this value

    Returns
    -------
    omega_ridge : (N_k,) dominant ω per k  (NaN if no peak)
    """
    i_min = np.searchsorted(omega_band, omega_min)
    omega_ridge = np.full(len(k_pos), np.nan)
    for i in range(len(k_pos)):
        row = spec_band[i, i_min:]
        if row.max() > 0:
            omega_ridge[i] = omega_band[i_min + int(np.argmax(row))]
    return omega_ridge


def plot_pinn_vs_bloch_comparison(
    k_pos,
    omega_pos,
    spectrum,
    k_coupling,
    mx,
    my=0.3,
    k_int_bloch=None,
    omega_min=0.01,
    omega_max=None,
    figsize=(7.2, 5.2),
):
    """
    Compare PINN-based ridge (from FFT spectrum) against Bloch-based curves.

    Bloch overlays:
      1) monoatomic branch (host chain),
      2) mass-in-mass acoustic/optical branches if k_int_bloch is provided.
    """
    k_ridge, omega_ridge = extract_ridge(k_pos, omega_pos, spectrum, omega_min=omega_min)
    k_line = np.linspace(0.0, np.pi, 400)

    mono = linear_dispersion(k_line, k_coupling, mx)
    out = {
        'k_ridge': k_ridge,
        'omega_ridge': omega_ridge,
        'k_line': k_line,
        'omega_mono': mono,
        'omega_mim_lo': None,
        'omega_mim_hi': None,
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_line / np.pi, mono, 'k--', lw=1.8, label='Bloch monoatomic')
    ax.plot(k_ridge / np.pi, omega_ridge, 'o', ms=4.0, color='tab:blue', alpha=0.9,
            label='PINN ridge (FFT)')

    if k_int_bloch is not None:
        om_lo, om_hi = mass_in_mass_dispersion(k_line, k_coupling, mx, my, k_int_bloch)
        out['omega_mim_lo'] = om_lo
        out['omega_mim_hi'] = om_hi
        ax.plot(k_line / np.pi, om_lo, color='tab:orange', lw=2.0,
                label=f'Bloch acoustic (k_int={k_int_bloch:.3g})')
        ax.plot(k_line / np.pi, om_hi, color='tab:red', lw=2.0,
                label='Bloch optical')

    if omega_max is None:
        omega_max = float(omega_pos[-1]) if len(omega_pos) else 1.0
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, omega_max)
    ax.set_xlabel(r'Wavenumber $k/\pi$')
    ax.set_ylabel(r'Frequency $\omega$ (rad/s)')
    ax.grid(alpha=0.25)
    ax.legend(framealpha=0.75)
    fig.tight_layout()
    return fig, ax, out


def estimate_impact_impulses_from_pinn(
    t_uniform,
    X_uniform,
    mx,
    accel_percentile=99.0,
    min_peak_distance=3,
    pre_samples=1,
    post_samples=1,
):
    """
    Estimate per-impact impulse from PINN displacement histories.

    Data-driven approximation based on velocity jumps around acceleration spikes:

        J ≈ m_x * (v_after - v_before)

    The output can guide the `p` selection in impact-branch reduced models.
    """
    t_uniform = np.asarray(t_uniform, dtype=float)
    X_uniform = np.asarray(X_uniform, dtype=float)
    if X_uniform.ndim != 2:
        raise ValueError('X_uniform must be 2D with shape (n_dof, n_time).')

    dt = float(t_uniform[1] - t_uniform[0])
    vel = np.gradient(X_uniform, dt, axis=1)
    acc = np.gradient(vel, dt, axis=1)

    impulses = []
    times = []
    dof_ids = []

    for i in range(X_uniform.shape[0]):
        a_abs = np.abs(acc[i])
        thr = np.percentile(a_abs, accel_percentile)
        peaks, _ = find_peaks(a_abs, height=thr, distance=min_peak_distance)

        for idx in peaks:
            i0 = max(idx - pre_samples, 0)
            i1 = min(idx + post_samples, X_uniform.shape[1] - 1)
            dv = vel[i, i1] - vel[i, i0]
            J = mx * dv
            impulses.append(J)
            times.append(t_uniform[idx])
            dof_ids.append(i)

    impulses = np.asarray(impulses, dtype=float)
    impulses_abs = np.abs(impulses)

    if impulses_abs.size == 0:
        p_recommended = 0.0
        p_mean_abs = 0.0
    else:
        p_recommended = float(np.median(impulses_abs))
        p_mean_abs = float(np.mean(impulses_abs))

    return {
        'impulses': impulses,
        'impulses_abs': impulses_abs,
        'times': np.asarray(times, dtype=float),
        'dof_ids': np.asarray(dof_ids, dtype=int),
        'p_recommended': p_recommended,
        'p_mean_abs': p_mean_abs,
        'n_events': int(impulses_abs.size),
        'accel_percentile': float(accel_percentile),
    }
