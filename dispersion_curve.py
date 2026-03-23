"""
dispersion_curve.py
-------------------
Stage 2: Nonlinear dispersion curve of a 1-D meta-impactor lattice.

What is a dispersion curve?
----------------------------
A dispersion curve plots the relationship between:

  ω  — angular frequency (rad/s), how fast a wave oscillates in TIME
  k  — wavenumber (rad / unit cell), how many wave cycles fit in SPACE

  ω = ω(k)  is the dispersion relation.

For a 1-D ring (periodic) lattice with N unit cells, the allowed wavenumbers are:

  k_n = 2π·n / N,   n = 0, 1, …, N/2                  (Brillouin zone boundary at k = π)

The linear (no impactor) acoustic branch for a monoatomic chain is:

  ω_lin(k) = 2·√(K/M)·|sin(k/2)|                       (1)

where K is the inter-cell spring stiffness and M is the primary mass.

With impact dampers present the dispersion becomes NONLINEAR and
AMPLITUDE-DEPENDENT: the curve shifts with excitation level.

Algorithm (2-D FFT method — the correct approach)
--------------------------------------------------
1.  Simulate a ring of N coupled meta-impactor unit cells for time T.
    Each cell n has primary mass M (spring-coupled to neighbours) and
    free-flying impactor m (impacts when |x_n − y_n| = D).
    Equations of motion BETWEEN impacts:

        M ẍ_n + c ẋ_n + K(2 x_n − x_{n-1} − x_{n+1}) = 0          (2)
        m ÿ_n = 0   (free flight)

    At each impact in cell n: velocity update via momentum + restitution.

2.  Record x_n(t): shape (N, T_steps) — the spatiotemporal field.

3.  Apply 2-D FFT in space (n → k) and time (t → ω):

        S(k, ω) = |FFT_2D [x_n(t)]|²                                (3)

4.  Plot S(k, ω) as a heatmap.  The dispersion relation ω(k) appears
    as bright ridges.

Usage
-----
from dispersion_curve import simulate_lattice, dispersion_from_2dfft, plot_dispersion_heatmap

t, x_nt = simulate_lattice(N=32, mx=1.0, my=0.3, k_coupling=1.0, D=1.0, T_sim=80.0)
k_vals, omega_vals, spectrum = dispersion_from_2dfft(t, x_nt)
plot_dispersion_heatmap(k_vals, omega_vals, spectrum, k_coupling=1.0, mx=1.0)
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Lattice simulation  (RK4 + event-based impact detection)
# ---------------------------------------------------------------------------

def simulate_lattice(
    N,
    mx,
    my,
    k_coupling,
    c=0.0,
    D=1.0,
    r=1.0,
    T_sim=80.0,
    n_steps=8000,
    ic_type='random',
    yt0_magnitude=1.0,
    seed=0,
):
    """
    Simulate a 1-D ring (periodic boundary conditions) of N meta-impactor
    unit cells.

    Equations of motion (between impacts)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Primary mass n:
        M ẍ_n = −K(2 x_n − x_{n−1} − x_{n+1}) − c ẋ_n
    Impactor n:
        m ÿ_n = 0   (free flight)

    At impact  |x_n − y_n| = D:
        [M, m; 1, −1] · v⁺ = [M, m; −r, r] · v⁻

    Periodic boundary: x_0 ≡ x_N, x_{N+1} ≡ x_1.

    Parameters
    ----------
    N             : int    — number of unit cells
    mx            : float  — primary mass M
    my            : float  — impactor mass m
    k_coupling    : float  — inter-cell spring stiffness K
    c             : float  — damping coefficient
    D             : float  — impact gap
    r             : float  — coefficient of restitution
    T_sim         : float  — total simulation time
    n_steps       : int    — number of time steps (dt = T_sim / n_steps)
    ic_type       : str
        'random'  — small random perturbation of primary masses (broadband,
                    excites all wavenumbers simultaneously → full dispersion)
        'impulse' — unit displacement on cell 0 only
        'wave'    — plane wave x_n(0) = A·cos(2π·n/N) (single wavenumber)
    yt0_magnitude : float  — initial impactor speed (same for all cells)
    seed          : int    — random seed (for 'random' IC only)

    Returns
    -------
    t        : (n_steps+1,) time array
    x_nt     : (N, n_steps+1) spatiotemporal primary-mass displacements
    """
    rng = np.random.default_rng(seed)
    dt  = T_sim / n_steps

    # Initial conditions
    x  = np.zeros(N)
    xt = np.zeros(N)
    y  = np.zeros(N)
    yt = np.full(N, -float(yt0_magnitude))   # all impactors start moving downward

    if ic_type == 'random':
        x  = rng.uniform(-0.05, 0.05, N)     # broadband spatial perturbation
    elif ic_type == 'impulse':
        x[0] = D * 0.4                       # small impulse on cell 0
    elif ic_type == 'wave':
        n_idx = np.arange(N)
        x = 0.1 * np.cos(2 * np.pi * n_idx / N)
    else:
        raise ValueError(f"Unknown ic_type: '{ic_type}'")

    # Storage
    x_history      = np.empty((N, n_steps + 1))
    x_history[:, 0] = x.copy()

    # Pre-build impact-update matrix (same for all cells)
    A_imp = np.array([[mx, my], [1.0, -1.0]])
    B_imp = np.array([[mx, my], [-r,   r  ]])

    # RK4 time integration
    for step in range(n_steps):
        def _f(x_, xt_):
            """Equations of motion for primary masses (periodic ring)."""
            x_left  = np.roll(x_,  1)   # x_{n-1}
            x_right = np.roll(x_, -1)   # x_{n+1}
            xtt = (-(k_coupling * (2.0 * x_ - x_left - x_right)
                   + c * xt_)) / mx
            return xt_, xtt

        k1v, k1a = _f(x,                   xt)
        k2v, k2a = _f(x + 0.5*dt*k1v, xt + 0.5*dt*k1a)
        k3v, k3a = _f(x + 0.5*dt*k2v, xt + 0.5*dt*k2a)
        k4v, k4a = _f(x +     dt*k3v, xt +     dt*k3a)

        x  += (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        xt += (dt / 6.0) * (k1a + 2*k2a + 2*k3a + k4a)
        y  += dt * yt                               # free flight

        # Impact detection and velocity update
        gap = x - y
        hit = np.where(np.abs(gap) >= D)[0]
        for n in hit:
            v_minus = np.array([xt[n], yt[n]])
            v_plus  = np.linalg.solve(A_imp, B_imp @ v_minus)
            xt[n]   = v_plus[0]
            yt[n]   = v_plus[1]

        x_history[:, step + 1] = x.copy()

    t = np.linspace(0.0, T_sim, n_steps + 1)
    return t, x_history


# ---------------------------------------------------------------------------
# 2-D FFT dispersion extraction
# ---------------------------------------------------------------------------

def dispersion_from_2dfft(t, x_nt, skip_transient=0.25):
    """
    Extract the frequency–wavenumber spectrum from spatiotemporal data.

    Parameters
    ----------
    t             : (T,) time array
    x_nt          : (N, T) primary-mass displacement array
    skip_transient: float in [0, 1), fraction of T to discard at the start
                    (removes startup transient before taking FFT)

    Returns
    -------
    k_pos     : (N//2+1,) positive wavenumbers in [0, π]  (rad / unit cell)
    omega_pos : (T//2+1,) positive angular frequencies     (rad / s)
    spectrum  : (N//2+1, T//2+1) one-sided power spectrum  |FFT_2D|²
    """
    N, T = x_nt.shape
    dt   = float(t[1] - t[0])

    # Drop the transient portion
    i0 = int(skip_transient * T)
    data = x_nt[:, i0:]
    T2   = data.shape[1]

    # 2-D FFT  (space axis 0 → k,  time axis 1 → ω)
    F    = np.fft.fft2(data)

    # Wavenumber axis: [0, 2π·(N-1)/N], keep positive half
    k_all  = 2.0 * np.pi * np.fft.fftfreq(N)     # rad / unit cell
    k_pos  = k_all[:N // 2 + 1]                   # [0, π]
    k_pos  = np.abs(k_pos)                         # ensure positive

    # Frequency axis: keep positive half
    om_all  = 2.0 * np.pi * np.fft.fftfreq(T2, d=dt)
    om_pos  = om_all[:T2 // 2 + 1]

    # One-sided spectrum (fold negative-k and negative-ω energy into positive side)
    # Simple approach: take the magnitude of the full FFT, then keep positive quadrant
    spectrum_full = np.abs(F) ** 2
    spectrum      = spectrum_full[:N // 2 + 1, :T2 // 2 + 1]

    return k_pos, om_pos, spectrum


# ---------------------------------------------------------------------------
# Linear dispersion  (reference)
# ---------------------------------------------------------------------------

def linear_dispersion(k_wavenumber, k_coupling, mx):
    """
    Acoustic branch of a monoatomic ring chain (no impactor, no damping):

        ω_lin(k) = 2 · √(K/M) · |sin(k/2)|

    Parameters
    ----------
    k_wavenumber : array-like, wavenumbers in [0, π]  (rad / unit cell)
    k_coupling   : float, inter-cell spring stiffness K
    mx           : float, primary mass M

    Returns
    -------
    omega_lin : ndarray
    """
    k = np.asarray(k_wavenumber, dtype=float)
    return 2.0 * np.sqrt(k_coupling / mx) * np.abs(np.sin(k / 2.0))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_dispersion_heatmap(
    k_pos,
    omega_pos,
    spectrum,
    k_coupling,
    mx,
    dB_range=40.0,
    save_path=None,
    figsize=(7, 5),
    omega_max=None,
):
    """
    Plot |FFT_2D(x_n(t))|² as a (k, ω) heatmap with the linear dispersion overlay.

    Parameters
    ----------
    k_pos, omega_pos, spectrum : outputs of dispersion_from_2dfft()
    k_coupling, mx : lattice parameters (for the linear baseline)
    dB_range       : dynamic range to display in dB (default 40 dB)
    save_path      : optional file path to save the figure
    figsize        : (w, h) in inches
    omega_max      : clip the ω axis at this value (None = auto)

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

    # Clip ω axis
    if omega_max is None:
        om_lin_max = linear_dispersion(np.pi, k_coupling, mx)
        omega_max  = 1.5 * om_lin_max
    i_om_max = np.searchsorted(omega_pos, omega_max)
    omega_plot = omega_pos[:i_om_max]
    spec_plot  = spectrum[:, :i_om_max]

    # Convert to dB and apply dynamic range
    S_dB = 10.0 * np.log10(spec_plot + 1e-30)
    S_dB -= S_dB.max()
    S_dB  = np.clip(S_dB, -dB_range, 0.0)

    # Normalised wavenumber for the x-axis
    k_norm = k_pos / np.pi     # [0, 1]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        k_norm, omega_plot, S_dB.T,
        cmap='inferno', shading='auto',
        vmin=-dB_range, vmax=0.0,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Spectral power (dB)', fontsize=FS - 4)
    cbar.ax.tick_params(labelsize=FS - 6)

    # Linear dispersion overlay
    k_line   = np.linspace(0, np.pi, 300)
    om_line  = linear_dispersion(k_line, k_coupling, mx)
    ax.plot(k_line / np.pi, om_line, 'w--', linewidth=LW,
            label=r'Linear  ($D \to \infty$)')

    ax.set_xlabel(r'Wavenumber  $k / \pi$  (–)',
                  fontsize=FS, labelpad=8)
    ax.set_ylabel(r'Frequency  $\omega$  (rad/s)',
                  fontsize=FS, labelpad=10)
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


def plot_dispersion_amplitude(
    results_by_amplitude,
    k_coupling,
    mx,
    save_path=None,
    figsize=(7, 5),
    omega_max=None,
):
    """
    Overlay dispersion ridge extractions for several excitation amplitudes.

    Parameters
    ----------
    results_by_amplitude : list of dicts, each with keys:
        'k'     : (M,) wavenumber array (rad/unit cell)
        'omega' : (M,) dominant frequency extracted per wavenumber
        'label' : str, e.g. 'A=0.5'
    k_coupling, mx : lattice parameters
    save_path, figsize, omega_max : as in plot_dispersion_heatmap

    Returns
    -------
    fig, ax
    """
    mpl.rcParams.update({
        'font.family':      'Times New Roman',
        'pdf.fonttype':     42,
        'ps.fonttype':      42,
    })
    FS      = 22
    LW      = 2.5
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, len(results_by_amplitude)))

    if omega_max is None:
        omega_max = 1.6 * linear_dispersion(np.pi, k_coupling, mx)

    fig, ax = plt.subplots(figsize=figsize)

    # Linear baseline
    k_line  = np.linspace(0, np.pi, 300)
    om_line = linear_dispersion(k_line, k_coupling, mx)
    ax.plot(k_line / np.pi, om_line, 'k--', linewidth=LW,
            label=r'Linear  ($D \to \infty$)')

    for res, color in zip(results_by_amplitude, palette):
        valid = ~np.isnan(res['omega'])
        ax.plot(res['k'][valid] / np.pi, res['omega'][valid],
                color=color, linewidth=LW,
                marker='o', markersize=5, label=res['label'])

    ax.set_xlabel(r'Wavenumber  $k / \pi$',     fontsize=FS, labelpad=8)
    ax.set_ylabel(r'Frequency  $\omega$ (rad/s)', fontsize=FS, labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, omega_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    ax.tick_params(axis='both', labelsize=FS - 2)
    ax.legend(fontsize=FS - 5, loc='upper left', framealpha=0.85)

    plt.tight_layout(pad=1.5)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {save_path}')

    plt.show()
    return fig, ax


def extract_ridge(k_pos, omega_pos, spectrum, omega_min=0.01):
    """
    Extract the dispersion ridge (dominant ω for each k) from the 2-D spectrum.

    Useful for overlaying multiple amplitude curves on one plot.

    Parameters
    ----------
    k_pos, omega_pos, spectrum : from dispersion_from_2dfft()
    omega_min : ignore frequencies below this (avoids DC peak)

    Returns
    -------
    k_pos   : same as input
    omega_ridge : (N//2+1,) dominant ω at each k
    """
    i_min  = np.searchsorted(omega_pos, omega_min)
    omega_ridge = np.full(len(k_pos), np.nan)
    for i in range(len(k_pos)):
        row = spectrum[i, i_min:]
        if row.max() > 0:
            idx = int(np.argmax(row)) + i_min
            omega_ridge[i] = omega_pos[idx]
    return k_pos, omega_ridge
