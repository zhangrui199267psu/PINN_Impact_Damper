"""
dispersion_theory.py
====================
Analytical dispersion relations and band-gap formulas for:
  - Monoatomic spring-mass chain
  - Mass-in-mass (locally resonant metamaterial) chain

All functions are pure NumPy — no TensorFlow or PINN required.

Physical setup (mass-in-mass chain)
-------------------------------------
Each unit cell has:
  - Primary mass  mx, coupled to neighbours via springs K_coupling
  - Internal mass my, attached to mx via an internal spring k_int

Bloch-reduced equations of motion (per unit cell):
    [K_b + k_int - ω²·mx,  -k_int          ] [X]   [0]
    [-k_int,               k_int - ω²·my   ] [Y] = [0]

where  K_b(k) = 2·K_coupling·(1 - cos(k))  is the Bloch spring stiffness.

Setting det = 0 gives a quadratic in ω²:
    mx·my·ω⁴ - [k_int·mx + (K_b + k_int)·my]·ω² + K_b·k_int = 0
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Monoatomic chain
# ---------------------------------------------------------------------------

def monoatomic_dispersion(k, K_coupling, mx):
    """
    Dispersion of a monoatomic spring-mass chain.

        ω(k) = 2·√(K_coupling / mx) · |sin(k / 2)|

    Parameters
    ----------
    k : array-like, shape (N,)
        Wavenumber in [0, π]  rad / unit-cell.
    K_coupling : float
        Inter-cell spring stiffness.
    mx : float
        Mass of each bead.

    Returns
    -------
    omega : ndarray, shape (N,)
        Angular frequency.
    """
    k = np.asarray(k, dtype=float)
    return 2.0 * np.sqrt(K_coupling / mx) * np.abs(np.sin(k / 2.0))


# ---------------------------------------------------------------------------
# Mass-in-mass (locally resonant) chain
# ---------------------------------------------------------------------------

def mass_in_mass_dispersion(k, K_coupling, mx, my, k_int):
    """
    Two dispersion branches of the linear mass-in-mass chain.

    The secular equation is quadratic in ω²:
        A·ω⁴ + B·ω² + C = 0
    with
        A = mx·my
        B = -[k_int·mx + (K_b + k_int)·my]
        C = K_b·k_int
        K_b(k) = 2·K_coupling·(1 - cos(k))

    Parameters
    ----------
    k : array-like, shape (N,)
    K_coupling, mx, my, k_int : float

    Returns
    -------
    omega_lower, omega_upper : ndarray, shape (N,)
        Acoustic (lower) and optical (upper) branch angular frequencies.
        At any k, the band gap lies between the two branches.
    """
    k = np.asarray(k, dtype=float)
    K_b = 2.0 * K_coupling * (1.0 - np.cos(k))          # Bloch stiffness

    A = mx * my
    B = -(k_int * mx + (K_b + k_int) * my)
    C = K_b * k_int

    discriminant = np.maximum(B**2 - 4.0 * A * C, 0.0)  # clip float noise
    omega2_lower = (-B - np.sqrt(discriminant)) / (2.0 * A)
    omega2_upper = (-B + np.sqrt(discriminant)) / (2.0 * A)

    # Ensure non-negative before sqrt (numerical safety)
    omega_lower = np.sqrt(np.maximum(omega2_lower, 0.0))
    omega_upper = np.sqrt(np.maximum(omega2_upper, 0.0))
    return omega_lower, omega_upper


def band_gap_edges(K_coupling, mx, my, k_int):
    """
    Analytical band-gap lower and upper edges for the mass-in-mass chain.

    The gap is widest at k = π where K_b = 4·K_coupling.
    Edges are evaluated at that k:
        ω_gap_lo  =  √(k_int / my)                   (internal resonance)
        ω_gap_hi  =  √(k_int·(mx + my) / (mx·my))   (upper gap edge)

    Parameters
    ----------
    K_coupling, mx, my, k_int : float

    Returns
    -------
    omega_lo, omega_hi : float
        Lower and upper band-gap edge frequencies (rad/s).
    """
    omega_lo = np.sqrt(k_int / my)
    omega_hi = np.sqrt(k_int * (mx + my) / (mx * my))
    return float(omega_lo), float(omega_hi)


def resonance_freq(k_int, my):
    """
    Internal resonance frequency  ω_res = √(k_int / my).

    This is the frequency at which the internal mass oscillates freely.
    It also marks the lower edge of the band gap.
    """
    return float(np.sqrt(k_int / my))


def estimate_k_int(omega_flat_band, my):
    """
    Estimate the effective internal spring constant from an observed flat-band
    (local resonance) frequency in the PINN spectrum.

        k_int_eff ≈ my · ω_flat_band²

    This inverts  ω_res = √(k_int / my).

    Parameters
    ----------
    omega_flat_band : float
        Frequency of the flat (dispersionless) band in the PINN dispersion plot.
    my : float
        Internal mass.

    Returns
    -------
    k_int_eff : float
    """
    return float(my * omega_flat_band**2)


# ---------------------------------------------------------------------------
# Group velocity
# ---------------------------------------------------------------------------

def group_velocity(k_vals, omega_vals):
    """
    Numerical group velocity  v_g = dω/dk  via central finite differences.

    Parameters
    ----------
    k_vals : array-like, shape (N,)
        Uniformly-spaced wavenumber array.
    omega_vals : array-like, shape (N,)
        Corresponding angular frequencies.

    Returns
    -------
    v_g : ndarray, shape (N,)
        Group velocity (same units as ω/k).
    """
    k_vals   = np.asarray(k_vals,   dtype=float)
    omega_vals = np.asarray(omega_vals, dtype=float)
    return np.gradient(omega_vals, k_vals)


# ---------------------------------------------------------------------------
# Impact-damper chain (appendix-inspired reduced model)
# ---------------------------------------------------------------------------

def impact_damper_branches(
    k,
    p,
    m=1.0,
    k_coupling=1.0,
    mu=0.3,
    p_ref=1.0,
    n_subharmonic=10,
):
    """
    Appendix-inspired dispersion branches for an impact-damper lattice.

    This helper follows the qualitative trends in
    "Dispersion relation - impactors in linear chain" (A9/A10 discussion):
      - bare branch: linear monoatomic chain,
      - acoustic branch: tends to 0 at low impact momentum p,
        tends to bare branch at high p,
      - optical branch: tends to bare branch at low p,
        approaches a high-frequency asymptote at high p,
      - subharmonic branches: odd-ratio family (1:3, 1:5, ...).

    Parameters
    ----------
    k : array-like
        Wavenumber in [0, pi].
    p : float
        Impact momentum-like control parameter.
    m, k_coupling : float
        Host chain mass and nearest-neighbour stiffness.
    mu : float
        Internal-to-primary mass ratio (reference uses mu=0.3).
    p_ref : float
        Blending scale for low/high-p transition.
    n_subharmonic : int
        Number of odd subharmonic branches to generate.

    Returns
    -------
    dict
        {
          'omega_bare', 'omega_acoustic', 'omega_optical', 'omega_subharmonics'
        }
    """
    k = np.asarray(k, dtype=float)
    omega_bare = monoatomic_dispersion(k, K_coupling=k_coupling, mx=m)

    # Smooth transition factor: 0 (low p) -> 1 (high p)
    alpha = float(np.clip(p / (p + p_ref + 1e-12), 0.0, 1.0))

    # Reference high-frequency scale from mass-in-mass resonance
    m_internal = max(mu * m, 1e-12)
    omega_hi = np.sqrt(k_coupling * (m + m_internal) / (m * m_internal))

    omega_acoustic = alpha * omega_bare
    omega_optical = (1.0 - alpha) * omega_bare + alpha * omega_hi

    subharmonics = []
    for j in range(n_subharmonic):
        odd = 2 * j + 3  # 3, 5, 7, ...
        subharmonics.append(omega_acoustic / odd)

    return {
        'omega_bare': omega_bare,
        'omega_acoustic': omega_acoustic,
        'omega_optical': omega_optical,
        'omega_subharmonics': np.asarray(subharmonics),
    }


def plot_impact_damper_dispersion(
    p_values,
    n_k=400,
    m=1.0,
    k_coupling=1.0,
    mu=0.3,
    p_ref=1.0,
    n_subharmonic=10,
    figsize=(8, 6),
    save_path=None,
):
    """Plot bare + impact-damper dispersion curves for one or multiple p values."""
    k = np.linspace(0.0, np.pi, n_k)
    p_values = np.atleast_1d(np.asarray(p_values, dtype=float))

    fig, ax = plt.subplots(figsize=figsize)

    for p in p_values:
        branches = impact_damper_branches(
            k,
            p=p,
            m=m,
            k_coupling=k_coupling,
            mu=mu,
            p_ref=p_ref,
            n_subharmonic=n_subharmonic,
        )

        ax.plot(k / np.pi, branches['omega_bare'], color='red', lw=2.0,
                alpha=0.9, label='Bare chain' if p == p_values[0] else None)
        ax.plot(k / np.pi, branches['omega_acoustic'], color='saddlebrown',
                lw=2.0, alpha=0.9,
                label=f'Impact branches (p={p:g})' if p == p_values[0] else None)
        ax.plot(k / np.pi, branches['omega_optical'], color='saddlebrown',
                lw=2.0, alpha=0.9)

        for omega_sh in branches['omega_subharmonics']:
            ax.plot(k / np.pi, omega_sh, color='saddlebrown', lw=1.0, alpha=0.35)

    ax.set_xlabel(r'Wavenumber $k/\pi$')
    ax.set_ylabel(r'Frequency $\omega$ (rad/s)')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax
