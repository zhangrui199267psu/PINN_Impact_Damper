"""
bloch_dispersion.py
-------------------
Approach 1 — Classical Bloch analysis for a 1-D meta-impactor lattice.

Physical model
--------------
Infinite periodic ring.  Unit cell n:
  Primary mass M,  spring-coupled to neighbours (stiffness K)
  Free-flying impactor m inside a gap D

EOM between impacts
~~~~~~~~~~~~~~~~~~~
  M ẍ_n + c ẋ_n + K (2 x_n − x_{n-1} − x_{n+1}) = 0
  m ÿ_n = 0   (free flight)

Impact when |x_n − y_n| = D:
  [[M, m], [1, −1]] v⁺ = [[M, m], [−r, r]] v⁻

Bloch reduction
~~~~~~~~~~~~~~~
Applying the Bloch condition  x_{n±1}(t) = x_n(t) e^{±ik}  converts
the infinite-chain EOM to a SINGLE-CELL oscillator:

  M ẍ + c ẋ + K_b(k) x = 0,    K_b(k) = 2K(1 − cos k)

with the same impactor dynamics.  The linear natural frequency is:

  ω_lin(k) = √(K_b(k)/M) = 2√(K/M) |sin(k/2)|

Key assumption / limitation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Bloch condition assumes:
  1.  Infinite lattice (no boundary effects, no finite-size corrections)
  2.  Identical dynamics in every cell (all cells in lock-step)
  3.  The impact orbit is eventually periodic (self-consistent)

The NUMBER of impacts per cycle is NOT fixed a priori — it emerges from
the simulation.  This makes the method more general than analytical
approaches that assume exactly one impact per period, but the Bloch
periodicity assumption (all cells identical) can break down for strongly
nonlinear or chaotic regimes.

Algorithm
---------
For each wavenumber k and initial displacement amplitude A:
  1. Set K_b = 2K(1 − cos k).
  2. Simulate the 2-DOF system (x, y) using RK4 + event-based impacts.
  3. After discarding the startup transient, extract the dominant
     oscillation frequency ω via 1-D FFT of x(t).
  4. Record ω(k, A).

Plot the resulting curves ω(k) per amplitude → amplitude-dependent
nonlinear dispersion surface with band-gap evolution.

Author: Rui Zhang
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Core: single-cell Bloch oscillator
# ---------------------------------------------------------------------------

def simulate_bloch_cell(
    k_val,
    mx, my,
    K_spring,
    D,
    r,
    c=0.0,
    x0=None,
    xt0=0.0,
    y0=0.0,
    yt0_mag=0.5,
    T_sim=200.0,
    n_steps=20000,
):
    """
    Simulate the Bloch-reduced single-cell impact oscillator.

    EOM (between impacts):
        mx * xtt + c * xt + K_b(k) * x = 0
        my * ytt = 0                         (free flight)

    where  K_b(k) = 2 * K_spring * (1 - cos(k_val))

    Impact when |x − y| ≥ D:
        [[mx, my], [1, -1]] · v⁺ = [[mx, my], [−r, r]] · v⁻

    Parameters
    ----------
    k_val    : float — wavenumber in [0, π]
    mx, my   : float — primary and impactor masses
    K_spring : float — inter-cell spring stiffness
    D        : float — impact gap
    r        : float — coefficient of restitution  (0 = fully plastic, 1 = elastic)
    c        : float — viscous damping on primary (default 0)
    x0       : float or None — initial displacement of primary (None → small random)
    xt0      : float — initial velocity of primary
    y0       : float — initial position of impactor
    yt0_mag  : float — initial speed of impactor (moves downward: yt = −yt0_mag)
    T_sim    : float — total simulation time (s)
    n_steps  : int   — number of RK4 steps

    Returns
    -------
    t      : (n_steps+1,) uniform time array
    x_hist : (n_steps+1,) primary displacement
    y_hist : (n_steps+1,) impactor displacement
    """
    K_b = 2.0 * K_spring * (1.0 - np.cos(k_val))
    dt  = T_sim / n_steps

    x  = float(x0) if x0 is not None else 0.05
    xt = float(xt0)
    y  = float(y0)
    yt = -abs(float(yt0_mag))    # impactor starts moving downward

    A_imp = np.array([[mx, my], [1.0, -1.0]])
    B_imp = np.array([[mx, my], [-r,   r  ]])

    x_hist = np.empty(n_steps + 1)
    y_hist = np.empty(n_steps + 1)
    x_hist[0], y_hist[0] = x, y

    for step in range(n_steps):
        # RK4 for primary (impactor has constant velocity in each sub-step)
        def _f(xi, xti):
            return xti, (-K_b * xi - c * xti) / mx

        k1v, k1a = _f(x, xt)
        k2v, k2a = _f(x + 0.5*dt*k1v, xt + 0.5*dt*k1a)
        k3v, k3a = _f(x + 0.5*dt*k2v, xt + 0.5*dt*k2a)
        k4v, k4a = _f(x +     dt*k3v, xt +     dt*k3a)

        x  += (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        xt += (dt / 6.0) * (k1a + 2*k2a + 2*k3a + k4a)
        y  += dt * yt                         # free flight

        # Impact detection
        if abs(x - y) >= D:
            v_minus = np.array([xt, yt])
            v_plus  = np.linalg.solve(A_imp, B_imp @ v_minus)
            xt = float(v_plus[0])
            yt = float(v_plus[1])

        x_hist[step + 1] = x
        y_hist[step + 1] = y

    t = np.linspace(0.0, T_sim, n_steps + 1)
    return t, x_hist, y_hist


# ---------------------------------------------------------------------------
# Frequency extraction from steady-state
# ---------------------------------------------------------------------------

def extract_frequency(t, x_hist, skip_transient=0.5, omega_min=0.01):
    """
    Extract the dominant angular frequency from the steady-state portion
    of the primary displacement history via 1-D FFT.

    Parameters
    ----------
    t              : (N,) uniform time array
    x_hist         : (N,) primary displacement
    skip_transient : fraction of T_sim to discard before FFT
    omega_min      : minimum angular frequency (rad/s) — avoids DC peak

    Returns
    -------
    omega     : dominant frequency (rad/s), or NaN if no clear peak
    amplitude : half-peak amplitude of the dominant mode
    """
    dt = float(t[1] - t[0])
    i0 = int(skip_transient * len(t))
    x  = x_hist[i0:] - x_hist[i0:].mean()   # remove DC offset

    if len(x) < 8:
        return np.nan, 0.0

    F   = np.fft.rfft(x)
    om  = 2.0 * np.pi * np.fft.rfftfreq(len(x), d=dt)

    i_min = max(1, np.searchsorted(om, omega_min))
    pwr   = np.abs(F[i_min:]) ** 2

    if pwr.max() == 0.0:
        return np.nan, 0.0

    i_peak = int(np.argmax(pwr)) + i_min
    amp    = float(np.abs(F[i_peak])) * 2.0 / len(x)
    return float(om[i_peak]), amp


# ---------------------------------------------------------------------------
# Bloch dispersion scan (multiple k and amplitudes)
# ---------------------------------------------------------------------------

def bloch_dispersion_scan(
    mx, my,
    K_spring,
    D, r,
    amplitudes,
    c=0.0,
    N_k=40,
    T_sim=300.0,
    n_steps=30000,
    yt0_mag=0.5,
    skip_transient=0.5,
    verbose=True,
):
    """
    Compute the nonlinear Bloch dispersion ω(k) for multiple initial amplitudes.

    For each (k, A) pair the single-cell Bloch oscillator is simulated.
    After the transient decays, the steady-state frequency is extracted
    by FFT and stored as ω(k, A).

    Parameters
    ----------
    mx, my      : primary and impactor masses
    K_spring    : inter-cell spring stiffness
    D           : impact gap
    r           : coefficient of restitution
    amplitudes  : list/array of initial primary-mass displacements A to scan
    c           : linear damping coefficient (default 0)
    N_k         : number of k points in [0, π]
    T_sim       : simulation time per (k, A) pair  — long enough for steady state
    n_steps     : number of RK4 time steps
    yt0_mag     : initial impactor speed
    skip_transient : fraction of T_sim to discard before FFT
    verbose     : print progress

    Returns
    -------
    k_vals  : (N_k,) wavenumber array in [0, π] rad/unit-cell
    results : list of dicts, one per amplitude:
        'A'     : initial amplitude
        'omega' : (N_k,) steady-state ω at each k  (NaN if no clear peak)
        'label' : plot label, e.g. 'A/D = 1.50'
    """
    k_vals     = np.linspace(0.0, np.pi, N_k)
    om_lin_arr = linear_dispersion(k_vals, K_spring, mx)
    results    = []

    for A in amplitudes:
        omega_arr = np.full(N_k, np.nan)
        if verbose:
            print(f'\n── A = {A:.3f}  (A/D = {A/D:.2f}) ──────────────────────')

        for j, k in enumerate(k_vals):
            if k == 0.0:
                # k=0: acoustic mode at ω=0 (rigid-body translation)
                omega_arr[j] = 0.0
                continue

            t_sim, x_h, _ = simulate_bloch_cell(
                k, mx, my, K_spring, D, r, c=c,
                x0=A, xt0=0.0, y0=0.0, yt0_mag=yt0_mag,
                T_sim=T_sim, n_steps=n_steps,
            )
            om, _ = extract_frequency(t_sim, x_h,
                                      skip_transient=skip_transient,
                                      omega_min=0.01)
            omega_arr[j] = om

            if verbose:
                om_ref = om_lin_arr[j]
                shift  = 100.0*(om - om_ref)/(om_ref + 1e-30) if not np.isnan(om) else float('nan')
                print(f'  k/π={k/np.pi:.3f}  ω={om:.4f}  ω_lin={om_ref:.4f}  Δω={shift:+.2f}%')

        results.append({
            'A':     A,
            'omega': omega_arr,
            'label': f'$A/D={A/D:.2f}$',
            'k':     k_vals,
        })

    return k_vals, results


# ---------------------------------------------------------------------------
# Linear dispersion reference
# ---------------------------------------------------------------------------

def linear_dispersion(k, K_spring, mx):
    """
    Acoustic branch of monoatomic ring chain (no impactor, no damping):
        ω_lin(k) = 2 √(K/M) |sin(k/2)|
    """
    return 2.0 * np.sqrt(K_spring / mx) * np.abs(np.sin(np.asarray(k) / 2.0))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_bloch_bands(
    k_vals, results,
    K_spring, mx,
    omega_max=None,
    save_path=None,
    figsize=(8, 5),
    title=None,
):
    """
    Overlay Bloch dispersion curves ω(k) for multiple initial amplitudes.

    Parameters
    ----------
    k_vals       : (N_k,) wavenumber array [0, π]
    results      : list of dicts from bloch_dispersion_scan()
    K_spring, mx : lattice parameters (for linear baseline)
    omega_max    : upper ω limit (default 1.5 × linear maximum)
    save_path    : optional file path to save figure
    figsize      : (w, h) inches
    title        : optional axes title string

    Returns
    -------
    fig, ax
    """
    mpl.rcParams.update({
        'font.family':      'Times New Roman',
        'mathtext.fontset': 'custom',
        'mathtext.rm':      'Times New Roman',
        'mathtext.it':      'Times New Roman:italic',
        'pdf.fonttype': 42,
        'ps.fonttype':  42,
    })
    FS = 20
    LW = 2.0

    om_lin_max = linear_dispersion(np.pi, K_spring, mx)
    if omega_max is None:
        omega_max = 1.5 * om_lin_max

    palette = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(results), 1)))

    fig, ax = plt.subplots(figsize=figsize)

    # Linear baseline
    k_line  = np.linspace(0, np.pi, 300)
    om_line = linear_dispersion(k_line, K_spring, mx)
    ax.plot(k_line / np.pi, om_line, 'k--', lw=LW,
            label=r'Linear  ($D\!\to\!\infty$)')

    for res, color in zip(results, palette):
        valid = ~np.isnan(res['omega']) & (res['omega'] <= omega_max * 1.05)
        if valid.any():
            ax.plot(k_vals[valid] / np.pi, res['omega'][valid],
                    'o-', color=color, lw=LW, ms=5,
                    label=res['label'])

    ax.set_xlabel(r'Wavenumber  $k/\pi$',          fontsize=FS, labelpad=8)
    ax.set_ylabel(r'Frequency  $\omega$  (rad/s)',  fontsize=FS, labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, omega_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    ax.tick_params(axis='both', labelsize=FS - 2)
    ax.legend(fontsize=FS - 5, loc='upper left', framealpha=0.85)
    if title:
        ax.set_title(title, fontsize=FS - 4)

    plt.tight_layout(pad=1.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()
    return fig, ax
