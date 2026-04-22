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
