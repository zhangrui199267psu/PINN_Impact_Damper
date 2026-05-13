"""
dispersion.py
-------------
Post-processing script: loads saved responses from newmark.py and/or pinn.py
and produces two sets of plots for each case:

  1. Per-DOF FFT  — normalised |FFT(x_i(t))| vs ω (rad/s), with theoretical
                    chain eigenfrequencies marked as vertical dotted lines.

  2. 2-D FFT dispersion map  — |FFT2(x(t,i))| in the (κ, ω) plane, with the
                               analytical dispersion relation overlaid.
                               Newmark and PINN side by side when both are
                               available.

Usage
-----
Run both solvers first, then:

    python dispersion.py                    # loads whatever is in Results_Linear_Chain/
    python dispersion.py --source newmark   # only Newmark data
    python dispersion.py --source pinn      # only PINN data

All figures saved to  Results_Linear_Chain/.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters  (must match newmark.py / pinn.py)
# ═══════════════════════════════════════════════════════════════════════════════

N_DOF  = 20
M_VAL  = 1.0
K_VAL  = 1.0

INPUT_VELOCITY_CASES = ['low', 'medium', 'high']
SELECTED_DOFS = [0, 4, 9, 14, 19]   # 0-indexed
SAVE_DIR = 'Results_Linear_Chain'
LATTICE_SPACING = 1.0                # d (m)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_case(source, case):
    """
    Load an NPZ file produced by newmark.py or pinn.py.

    Returns a dict with keys: t, x, xt, E, v_in
    Returns None if the file does not exist.
    """
    path = os.path.join(SAVE_DIR, f'{source}_{case}.npz')
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical helpers
# ═══════════════════════════════════════════════════════════════════════════════

def chain_eigenfreqs(n=N_DOF, m=M_VAL, k=K_VAL):
    """
    Exact eigenfrequencies for a free-free uniform chain:
        ω_j = 2√(k/m) |sin(j π / (2n))|   for j = 0, 1, ..., n-1
    j=0 is the rigid-body mode (ω=0).
    """
    j = np.arange(n)
    return 2.0 * np.sqrt(k / m) * np.abs(np.sin(j * np.pi / (2.0 * n)))


def analytical_dispersion(d=LATTICE_SPACING, m=M_VAL, k=K_VAL, n_pts=500):
    """Analytical dispersion relation ω(κ) = 2√(k/m)|sin(κd/2)|."""
    kappa = np.linspace(0.0, np.pi / d, n_pts)
    omega = 2.0 * np.sqrt(k / m) * np.abs(np.sin(kappa * d / 2.0))
    return kappa, omega


# ═══════════════════════════════════════════════════════════════════════════════
# 1-D FFT per DOF
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fft_per_dof(datasets, case, v_in, out_dir):
    """
    Plot the per-DOF frequency spectrum for one or two datasets.

    datasets : dict  {label: {'t': ..., 'x': ...}}
               e.g. {'Newmark-β': nm_data, 'PINN': pinn_data}

    For each DOF in SELECTED_DOFS:
      - Hann-windowed one-sided FFT
      - Normalised |FFT| vs ω (rad/s)
      - Theoretical eigenfrequencies as vertical dotted lines

    Saved as:  fft_per_dof_{case}.png
    """
    omega_eig = chain_eigenfreqs()
    omega_max = 2.0 * np.sqrt(K_VAL / M_VAL)

    colors = ['k', 'C1', 'C0', 'C3']
    styles = ['-', '--', '-.', ':']

    n_sel = len(SELECTED_DOFS)
    fig, axes = plt.subplots(n_sel, 1,
                             figsize=(11, 3 * n_sel),
                             sharex=True)

    for ax, di in zip(axes, SELECTED_DOFS):
        for idx, (label, data) in enumerate(datasets.items()):
            t = data['t']
            x = data['x']
            dt  = float(t[1] - t[0])
            n_t = len(t)
            win = np.hanning(n_t)

            freqs    = np.fft.rfftfreq(n_t, d=dt) * 2.0 * np.pi   # rad/s
            spectrum = np.abs(np.fft.rfft(x[:, di] * win))
            peak     = spectrum.max() or 1.0
            spectrum /= peak

            ax.plot(freqs, spectrum,
                    color=colors[idx % len(colors)],
                    ls=styles[idx % len(styles)],
                    lw=1.3, alpha=0.9, label=label)

        # mark eigenfrequencies (skip ω=0 rigid-body mode)
        for oe in omega_eig[1:]:
            ax.axvline(oe, color='C2', lw=0.7, ls=':', alpha=0.55)

        ax.set_ylabel(f'DOF {di+1}\n|FFT| (norm.)', fontsize=9)
        ax.set_ylim(0.0, 1.15)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('ω  (rad/s)')
    axes[-1].set_xlim(0.0, omega_max * 1.15)

    label_str = '  vs  '.join(datasets.keys())
    fig.suptitle(
        f'Per-DOF frequency spectrum  [{label_str}]\n'
        f'{case.capitalize()}  v0={v_in:.1f} m/s  '
        f'(green dotted lines = chain eigenfrequencies)',
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f'fft_per_dof_{case}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 2-D FFT dispersion map
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_2dfft(x_tn, dt, d=LATTICE_SPACING):
    """
    2-D FFT of a space-time response matrix.

    Parameters
    ----------
    x_tn : (n_time, n_dof)   displacement matrix (rows = time, cols = DOF)
    dt   : temporal sampling interval (s)
    d    : lattice spacing (m)

    Returns
    -------
    k     : (n_k,)     wavenumber array  κ ∈ [0, π/d]  (rad/m)
    omega : (n_ω,)     angular frequency ω ≥ 0          (rad/s)
    S     : (n_ω, n_k) normalised |FFT2|
    """
    x_tn = np.asarray(x_tn, dtype=float)
    n_t, n_x = x_tn.shape

    # subtract temporal mean, apply 2-D Hann window
    x0 = x_tn - np.mean(x_tn, axis=0, keepdims=True)
    wt = np.hanning(n_t)[:, None]
    wx = np.hanning(n_x)[None, :]
    xw = x0 * wt * wx

    F_full = np.fft.fft2(xw)
    S_full = np.abs(F_full)

    omega_raw = 2.0 * np.pi * np.fft.fftfreq(n_t, d=dt)
    k_raw     = 2.0 * np.pi * np.fft.fftfreq(n_x, d=d)

    om_mask = omega_raw >= 0.0
    k_mask  = (k_raw >= 0.0) & (k_raw <= np.pi / d + 1e-10)

    omega = omega_raw[om_mask]
    k     = k_raw[k_mask]
    S     = S_full[np.ix_(om_mask, k_mask)]

    if S.max() > 0:
        S = S / S.max()

    return k, omega, S


def plot_dispersion_2dfft(datasets, case, v_in, out_dir, d=LATTICE_SPACING):
    """
    Plot the 2-D FFT dispersion map for one or two datasets side by side.

    datasets : dict  {label: {'t': ..., 'x': ...}}

    Each panel shows the (κ, ω) intensity map with the analytical dispersion
    relation ω(κ) = 2√(k/m)|sin(κd/2)| overlaid as a white dashed line.

    Axes:
      x = κ / (π/d) ∈ [0, 1]   (normalised wavenumber)
      y = ω (rad/s)

    Saved as:  dispersion_2dfft_{case}.png
    """
    kappa_ana, omega_ana = analytical_dispersion(d)
    kpi_ana  = kappa_ana / (np.pi / d)
    omega_max = 2.0 * np.sqrt(K_VAL / M_VAL)

    n_panels = len(datasets)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(6 * n_panels, 5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, datasets.items()):
        t  = data['t']
        x  = data['x']
        dt = float(t[1] - t[0])

        k, omega, S = _compute_2dfft(x, dt, d)
        kpi = k / (np.pi / d)

        pcm = ax.pcolormesh(kpi, omega, S,
                            shading='auto', cmap='magma',
                            vmin=0.0, vmax=1.0)
        ax.plot(kpi_ana, omega_ana, 'w--', lw=1.8,
                label='Analytical ω(κ)')

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, omega_max * 1.15)
        ax.set_xlabel('κ / (π/d)  —  normalised wavenumber')
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.18, color='white', lw=0.5)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04,
                     label='|FFT2| (norm.)')

    axes[0].set_ylabel('ω  (rad/s)')
    label_str = '  vs  '.join(datasets.keys())
    fig.suptitle(
        f'2-D FFT dispersion map  [{label_str}]\n'
        f'{case.capitalize()}  v0={v_in:.1f} m/s',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f'dispersion_2dfft_{case}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Dispersion analysis from saved Newmark / PINN responses.'
    )
    parser.add_argument(
        '--source',
        choices=['newmark', 'pinn', 'both'],
        default='both',
        help='Which solver results to load (default: both)',
    )
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    sources_to_try = {
        'newmark': 'Newmark-β',
        'pinn':    'PINN',
    }
    if args.source == 'newmark':
        sources_to_try = {'newmark': 'Newmark-β'}
    elif args.source == 'pinn':
        sources_to_try = {'pinn': 'PINN'}

    for case in INPUT_VELOCITY_CASES:
        print(f'\n{"="*60}')
        print(f'  Case: {case}')
        print(f'{"="*60}')

        # load whatever is available
        datasets = {}
        v_in_case = None
        for src_key, label in sources_to_try.items():
            data = load_case(src_key, case)
            if data is None:
                print(f'  [{src_key}] file not found — skipping'
                      f' ({SAVE_DIR}/{src_key}_{case}.npz)')
                continue
            datasets[label] = data
            v_in_case = float(data['v_in'])
            print(f'  Loaded {SAVE_DIR}/{src_key}_{case}.npz'
                  f'  ({len(data["t"])} time steps)')

        if not datasets:
            print(f'  No data found for case {case} — skipping.')
            continue

        # ── per-DOF FFT ────────────────────────────────────────────────────────
        print('\n  [1] Per-DOF FFT')
        plot_fft_per_dof(datasets, case, v_in_case, SAVE_DIR)

        # ── 2-D FFT dispersion ─────────────────────────────────────────────────
        print('  [2] 2-D FFT dispersion map')
        plot_dispersion_2dfft(datasets, case, v_in_case, SAVE_DIR)

    print(f'\nDone. All figures saved to {SAVE_DIR}/')


if __name__ == '__main__':
    main()
