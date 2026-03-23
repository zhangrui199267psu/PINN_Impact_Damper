"""
dispersion_curve.py
-------------------
Stage 2: Compute the nonlinear dispersion curve of a 1-D meta-impactor lattice
using the parametric PINN trained in Stage 1.

Background
----------
For an infinite 1-D chain of primary masses m connected by springs k_c, each
carrying an internal impactor of mass μ, the Bloch boundary condition reduces
the lattice problem to a single-unit-cell SDOF system with effective stiffness

    k_eff(κ) = 2 · k_c · (1 − cos κ)                           (1)

where κ ∈ [0, π] is the wavenumber (normalised by the lattice spacing d = 1).

The linear dispersion (no impactor) is

    ω_lin(κ) = √(k_eff / m) = 2 · √(k_c / m) · |sin(κ/2)|     (2)

With impact dampers present the dispersion becomes amplitude-dependent and
nonlinear.  The parametric PINN trained in Stage 1 maps

    (t, m, μ, k, c, D) → x(t)

for any k within the training range.  Sweeping κ and querying the predictor
with k = k_eff(κ) gives the full time series; FFT extracts ω.

Usage
-----
from parametric_pinn_multi_case import train_parametric_pinn
from dispersion_curve import compute_dispersion_curve, plot_dispersion_curve

predictor, _ = train_parametric_pinn(...)        # Stage 1

kappa, curves = compute_dispersion_curve(
    predictor,
    k_coupling=1.0,
    mx=1.0, my=0.3,
    c=0.0, D=1.0,
    yt0_values=[-0.5, -1.0, -1.5],              # amplitude sweep
)
plot_dispersion_curve(kappa, curves, k_coupling=1.0, mx=1.0,
                      save_path='dispersion.png')
"""

import warnings
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Frequency extraction from a time series
# ---------------------------------------------------------------------------

def _extract_frequency_fft(t, x):
    """
    Dominant angular frequency via FFT.

    t, x are 1-D (possibly non-uniformly sampled: each segment has
    equal spacing but different durations, so the concatenation is
    piecewise-uniform).  We re-interpolate to a single uniform grid.

    Returns omega (rad / s), or np.nan if extraction fails.
    """
    t = t.flatten()
    x = x.flatten()
    if len(t) < 8:
        return np.nan

    # Re-interpolate to uniform time grid
    n_uniform = 4 * len(t)
    t_uni = np.linspace(t[0], t[-1], n_uniform)
    try:
        x_uni = interp1d(t, x, kind='linear', fill_value='extrapolate')(t_uni)
    except Exception:
        return np.nan

    dt = t_uni[1] - t_uni[0]
    x_uni -= np.mean(x_uni)                        # remove DC

    spectrum = np.abs(np.fft.rfft(x_uni))
    freqs    = np.fft.rfftfreq(len(x_uni), d=dt)   # cycles / s

    # Ignore DC (index 0)
    idx_dom  = int(np.argmax(spectrum[1:])) + 1
    omega    = 2.0 * np.pi * freqs[idx_dom]
    return float(omega)


def _extract_frequency_impact(impact_times):
    """
    Estimate frequency from mean inter-impact period.

    For a limit-cycle motion impacted once per oscillation cycle,
    the cycle period equals the mean inter-impact time.

    Returns omega (rad / s), or np.nan.
    """
    T = np.asarray(impact_times, dtype=float)
    if len(T) < 2:
        return np.nan
    T_mean = float(np.mean(T))
    return 2.0 * np.pi / T_mean


# ---------------------------------------------------------------------------
# Linear dispersion (reference)
# ---------------------------------------------------------------------------

def linear_dispersion(kappa, k_coupling, mx):
    """
    Acoustic dispersion of a monoatomic chain (no impactor, no damping).

        ω_lin(κ) = 2 · √(k_c / m) · |sin(κ/2)|

    Parameters
    ----------
    kappa      : array-like of wavenumbers in [0, π]
    k_coupling : float, k_c
    mx         : float, primary mass

    Returns
    -------
    omega_lin : ndarray
    """
    kappa = np.asarray(kappa, dtype=float)
    return 2.0 * np.sqrt(k_coupling / mx) * np.abs(np.sin(kappa / 2.0))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def compute_dispersion_curve(
    predictor,
    k_coupling,
    mx,
    my,
    c=0.0,
    D=1.0,
    yt0_values=(-1.0,),
    n_kappa=50,
    kappa_min=None,
    kappa_max=np.pi,
    num_points=1000,
    freq_method='fft',
    warn_bounds=True,
):
    """
    Sweep κ ∈ [kappa_min, kappa_max] and extract ω(κ) via the trained PINN.

    Parameters
    ----------
    predictor   : ParametricSequentialPredictor from train_parametric_pinn()
    k_coupling  : float  — inter-cell coupling stiffness k_c
    mx          : float  — primary mass per unit cell
    my          : float  — internal (impactor) mass per unit cell
    c           : float  — damping coefficient (same for all κ)
    D           : float  — impact gap
    yt0_values  : iterable of float — initial impactor velocities (one curve per value)
    n_kappa     : int    — number of wavenumber sample points
    kappa_min   : float  — minimum κ (defaults to π/n_kappa to avoid k_eff ≈ 0)
    kappa_max   : float  — maximum κ (default π, zone boundary)
    num_points  : int    — PINN time-series points per impact segment
    freq_method : {'fft', 'impact'}
                    'fft'    — FFT of x(t) (better for many segments)
                    'impact' — period from mean inter-impact time
    warn_bounds : bool   — warn when k_eff is outside predictor training range

    Returns
    -------
    kappa  : (n_kappa,) ndarray of wavenumbers
    curves : list of dicts, one per yt0 value.
             Each dict has keys:
               'kappa'   : (n_kappa,) wavenumber array
               'omega'   : (n_kappa,) extracted angular frequency
               'k_eff'   : (n_kappa,) effective stiffness
               'yt0'     : scalar, initial impactor velocity used
               'omega_lin': (n_kappa,) linear baseline for this sweep
    """
    if kappa_min is None:
        kappa_min = np.pi / n_kappa         # avoid κ = 0 (k_eff = 0)

    kappa   = np.linspace(kappa_min, kappa_max, n_kappa)
    k_eff   = 2.0 * k_coupling * (1.0 - np.cos(kappa))
    om_lin  = linear_dispersion(kappa, k_coupling, mx)

    # Attempt to read predictor training bounds (for out-of-range warning)
    try:
        # The first model's lb/ub store the training range
        lb_k = float(predictor.models[0].lb[0, 3])   # index 3 = k channel
        ub_k = float(predictor.models[0].ub[0, 3])
    except Exception:
        lb_k, ub_k = None, None

    curves = []

    for yt0 in yt0_values:
        print(f'\nSweeping κ for yt0 = {yt0:.2f}  |  {n_kappa} points')
        print(f'  k_eff range: [{k_eff.min():.4f}, {k_eff.max():.4f}]', end='')
        if lb_k is not None:
            print(f'  (training k: [{lb_k:.4f}, {ub_k:.4f}])', end='')
            if warn_bounds and (k_eff.min() < lb_k or k_eff.max() > ub_k):
                warnings.warn(
                    f"k_eff range [{k_eff.min():.3f}, {k_eff.max():.3f}] partly outside "
                    f"training range [{lb_k:.3f}, {ub_k:.3f}]. "
                    "Retrain with wider k bounds for accurate extrapolation.",
                    UserWarning, stacklevel=2,
                )
        print()

        omega_vals = np.full(n_kappa, np.nan)

        for i, (kap, keff) in enumerate(zip(kappa, k_eff)):
            try:
                res = predictor.predict(
                    mx=mx, my=my, k=float(keff), c=c, D=D,
                    x0=0.0, xt0=0.0, y0=0.0, yt0=float(yt0),
                    num_points=num_points,
                )
                if freq_method == 'fft':
                    omega = _extract_frequency_fft(res['time'], res['x'])
                elif freq_method == 'impact':
                    omega = _extract_frequency_impact(res['impact_times'])
                else:
                    raise ValueError(f"Unknown freq_method: '{freq_method}'")
            except Exception as exc:
                warnings.warn(f"κ={kap:.3f}: prediction failed — {exc}")
                omega = np.nan

            omega_vals[i] = omega

            if (i + 1) % max(1, n_kappa // 5) == 0 or i == 0:
                status = f'{omega:.4f}' if not np.isnan(omega) else 'NaN'
                print(f'  [{i+1:3d}/{n_kappa}]  κ={kap:.3f}  k_eff={keff:.3f}  ω={status}')

        curves.append({
            'kappa':    kappa,
            'omega':    omega_vals,
            'k_eff':    k_eff,
            'yt0':      float(yt0),
            'omega_lin': om_lin,
        })

    return kappa, curves


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dispersion_curve(
    kappa,
    curves,
    k_coupling,
    mx,
    normalize_kappa=True,
    save_path=None,
    figsize=(7, 5),
):
    """
    Plot dispersion curve(s) with the linear baseline.

    Parameters
    ----------
    kappa, curves   : outputs of compute_dispersion_curve()
    k_coupling, mx  : lattice parameters (for linear baseline label)
    normalize_kappa : if True, x-axis is κ/π ∈ [0, 1]; otherwise κ in rad
    save_path       : optional file path to save the figure
    figsize         : (width, height) in inches

    Returns
    -------
    fig, ax
    """
    mpl.rcParams.update({
        'font.family':   'Times New Roman',
        'mathtext.fontset': 'custom',
        'mathtext.rm':   'Times New Roman',
        'mathtext.it':   'Times New Roman:italic',
        'mathtext.bf':   'Times New Roman:bold',
        'pdf.fonttype':  42,
        'ps.fonttype':   42,
    })

    FS   = 22
    LW   = 2
    MS   = 5

    fig, ax = plt.subplots(figsize=figsize)

    # --- linear baseline (use first curve's omega_lin) ---
    om_lin = curves[0]['omega_lin']
    x_axis = kappa / np.pi if normalize_kappa else kappa
    ax.plot(x_axis, om_lin, 'k--', linewidth=LW,
            label=r'Linear  ($D\to\infty$)')

    # --- PINN nonlinear curves ---
    n_curves = len(curves)
    palette  = plt.cm.plasma(np.linspace(0.15, 0.85, max(n_curves, 1)))

    for res, color in zip(curves, palette):
        label = rf'PINN  $\dot{{y}}_0 = {res["yt0"]:.1f}$'
        valid = ~np.isnan(res['omega'])
        ax.plot(x_axis[valid], res['omega'][valid],
                color=color, linewidth=LW,
                marker='o', markersize=MS, label=label)

    # --- labels & formatting ---
    x_label = (r'Normalized wavenumber  $\kappa / \pi$'
               if normalize_kappa else r'Wavenumber  $\kappa$  (rad)')
    ax.set_xlabel(x_label,                  fontsize=FS, labelpad=8)
    ax.set_ylabel(r'Frequency  $\omega$  (rad/s)', fontsize=FS, labelpad=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    ax.tick_params(axis='both', labelsize=FS - 2)
    ax.legend(fontsize=FS - 5, loc='upper left', framealpha=0.85)

    if normalize_kappa:
        ax.set_xlim(0, 1)
    else:
        ax.set_xlim(0, np.pi)

    ax.set_ylim(bottom=0)
    plt.tight_layout(pad=1.5)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {save_path}')

    plt.show()
    return fig, ax


def plot_dispersion_with_keff(
    kappa,
    curves,
    k_coupling,
    save_path=None,
):
    """
    Two-panel figure: (left) ω(κ) dispersion, (right) k_eff(κ).
    Helpful for checking the Bloch stiffness used at each wavenumber.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: dispersion
    ax = axes[0]
    k_vals = curves[0]['k_eff']
    x_axis = kappa / np.pi

    om_lin = curves[0]['omega_lin']
    ax.plot(x_axis, om_lin, 'k--', lw=2, label=r'Linear')

    palette = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(curves), 1)))
    for res, color in zip(curves, palette):
        valid = ~np.isnan(res['omega'])
        ax.plot(x_axis[valid], res['omega'][valid],
                color=color, lw=2, marker='o', ms=4,
                label=rf"$\dot{{y}}_0={res['yt0']:.1f}$")

    ax.set_xlabel(r'$\kappa / \pi$', fontsize=18)
    ax.set_ylabel(r'$\omega$  (rad/s)', fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=13, loc='upper left')
    ax.set_title('Dispersion curve', fontsize=16)

    # Right: k_eff(κ)
    ax2 = axes[1]
    ax2.plot(x_axis, k_vals, 'b-', lw=2)
    ax2.set_xlabel(r'$\kappa / \pi$', fontsize=18)
    ax2.set_ylabel(r'$k_{\mathrm{eff}}(\kappa) = 2k_c(1-\cos\kappa)$', fontsize=16)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(labelsize=16)
    ax2.set_title('Bloch effective stiffness', fontsize=16)
    ax2.axhline(y=2 * k_coupling, color='gray', lw=1, linestyle=':', label=r'$k = 2k_c$')
    ax2.axhline(y=4 * k_coupling, color='gray', lw=1, linestyle='--', label=r'$k = 4k_c$')
    ax2.legend(fontsize=13)

    plt.tight_layout(pad=1.5)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()
    return fig, axes
