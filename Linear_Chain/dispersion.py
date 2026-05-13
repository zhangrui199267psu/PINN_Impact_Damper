"""
dispersion.py
-------------
Post-processing script: loads saved responses from newmark.py and/or pinn.py
and produces two sets of plots for each case:

  1. Per-DOF FFT  — energy spectrum |FFT(x_i)|² vs frequency (Hz) for every
                    DOF in SELECTED_DOFS.  Each DOF is a separate subplot.
                    Theoretical chain eigenfrequencies are marked.

  2. 2-D FFT dispersion map  — |FFT2(x(t,i))| in the (κ, ω) plane with the
                               analytical dispersion relation overlaid.
                               Newmark and PINN panels shown side by side
                               when both sources are available.

Folder structure
----------------
  Results_Linear_Chain/
  ├── Newmark/      newmark_{case}.npz   (written by newmark.py)
  ├── PINN/         pinn_{case}.npz      (written by pinn.py)
  └── Dispersion/   all figures from this script

Usage
-----
    python dispersion.py                    # loads both sources if present
    python dispersion.py --source newmark
    python dispersion.py --source pinn
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
SELECTED_DOFS        = [0, 4, 9, 14, 19]   # 0-indexed
LATTICE_SPACING      = 1.0                  # d (m)

BASE_DIR       = 'Results_Linear_Chain'
NEWMARK_DIR    = os.path.join(BASE_DIR, 'Newmark')
PINN_DIR       = os.path.join(BASE_DIR, 'PINN')
DISP_DIR       = os.path.join(BASE_DIR, 'Dispersion')

SOURCE_DIRS = {
    'newmark': NEWMARK_DIR,
    'pinn':    PINN_DIR,
}
SOURCE_LABELS = {
    'newmark': 'Newmark-β',
    'pinn':    'PINN',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_case(source_key, case):
    """
    Load Results_Linear_Chain/{Newmark|PINN}/{source}_{case}.npz.
    Returns dict of arrays, or None if the file does not exist.
    """
    path = os.path.join(SOURCE_DIRS[source_key], f'{source_key}_{case}.npz')
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical helpers
# ═══════════════════════════════════════════════════════════════════════════════

def chain_eigenfreqs_hz(n=N_DOF, m=M_VAL, k=K_VAL):
    """
    Eigenfrequencies in Hz for a free-free uniform chain:
        f_j = (1/2π) · 2√(k/m) |sin(jπ/(2n))|   j = 0…n-1
    j=0 is the rigid-body mode (f=0).
    """
    j = np.arange(n)
    omega_j = 2.0 * np.sqrt(k / m) * np.abs(np.sin(j * np.pi / (2.0 * n)))
    return omega_j / (2.0 * np.pi)


def analytical_dispersion(d=LATTICE_SPACING, m=M_VAL, k=K_VAL, n_pts=500):
    """
    Analytical dispersion relation:
        ω(κ) = 2√(k/m) |sin(κ d / 2)|
    Returns kappa (rad/m), omega (rad/s).
    """
    kappa = np.linspace(0.0, np.pi / d, n_pts)
    omega = 2.0 * np.sqrt(k / m) * np.abs(np.sin(kappa * d / 2.0))
    return kappa, omega


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — Per-DOF energy spectrum  (1-D FFT)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fft_per_dof(datasets, case, v_in, out_dir):
    """
    One subplot per DOF in SELECTED_DOFS.

    x-axis : frequency  f  (Hz)
    y-axis : energy spectrum  |FFT(x_i)|²  (m²)

    Theoretical chain eigenfrequencies are marked as vertical dotted lines.
    One figure per source; when two sources are present they share the same
    figure with overlaid lines so the spectra can be compared directly.

    Saved as:  fft_energy_per_dof_{case}.png
    """
    f_eig    = chain_eigenfreqs_hz()
    f_max_hz = (2.0 * np.sqrt(K_VAL / M_VAL)) / (2.0 * np.pi) * 1.15

    colors = ['k', 'C1', 'C0', 'C3']
    styles = ['-', '--', '-.', ':']

    n_sel = len(SELECTED_DOFS)
    fig, axes = plt.subplots(n_sel, 1,
                             figsize=(11, 3 * n_sel),
                             sharex=True)

    for ax, di in zip(axes, SELECTED_DOFS):
        for idx, (label, data) in enumerate(datasets.items()):
            t   = data['t']
            x   = data['x']
            dt  = float(t[1] - t[0])
            n_t = len(t)

            # Hann window to reduce spectral leakage
            win     = np.hanning(n_t)
            X       = np.fft.rfft(x[:, di] * win)
            freqs   = np.fft.rfftfreq(n_t, d=dt)          # Hz
            energy  = np.abs(X) ** 2                       # m²  (ESD)

            ax.plot(freqs, energy,
                    color=colors[idx % len(colors)],
                    ls=styles[idx % len(styles)],
                    lw=1.3, alpha=0.9, label=label)

        # mark eigenfrequencies (skip f=0 rigid-body mode)
        for fe in f_eig[1:]:
            ax.axvline(fe, color='C2', lw=0.8, ls=':', alpha=0.6)

        ax.set_ylabel(f'DOF {di+1}\n|X(f)|²  (m²)', fontsize=9)
        ax.set_yscale('log')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3, which='both')

    axes[-1].set_xlabel('Frequency  f  (Hz)')
    axes[-1].set_xlim(0.0, f_max_hz)

    label_str = '  vs  '.join(datasets.keys())
    fig.suptitle(
        f'Energy spectrum per DOF  [{label_str}]\n'
        f'{case.capitalize()}  v₀ = {v_in:.1f} m/s'
        f'   (green dotted = chain eigenfrequencies)',
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f'fft_energy_per_dof_{case}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — 2-D FFT dispersion map
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_2dfft(x_tn, dt, d=LATTICE_SPACING):
    """
    2-D FFT of a space-time displacement matrix.

    Parameters
    ----------
    x_tn : (n_time, n_dof)   rows = time, cols = DOF (spatial index)
    dt   : temporal sampling interval (s)
    d    : lattice spacing (m)

    Returns
    -------
    k     : (n_k,)     wavenumber κ ∈ [0, π/d]  (rad/m)
    omega : (n_ω,)     angular frequency ω ≥ 0   (rad/s)
    S     : (n_ω, n_k) normalised |FFT2|
    """
    x_tn = np.asarray(x_tn, dtype=float)
    n_t, n_x = x_tn.shape

    # subtract temporal mean; apply 2-D Hann window
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
    2-D FFT dispersion map — one panel per source.

    x-axis : normalised wavenumber  κ/(π/d) ∈ [0, 1]
    y-axis : angular frequency  ω  (rad/s)

    The analytical dispersion relation ω(κ)=2√(k/m)|sin(κd/2)| is overlaid
    as a white dashed curve on every panel.

    Saved as:  dispersion_2dfft_{case}.png
    """
    kappa_ana, omega_ana = analytical_dispersion(d)
    kpi_ana   = kappa_ana / (np.pi / d)
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
        ax.plot(kpi_ana, omega_ana, 'w--', lw=1.8, label='Analytical ω(κ)')

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
        f'{case.capitalize()}  v₀ = {v_in:.1f} m/s',
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
        description='Dispersion / FFT analysis from saved Newmark / PINN responses.'
    )
    parser.add_argument(
        '--source',
        choices=['newmark', 'pinn', 'both'],
        default='both',
        help='Which solver results to load (default: both)',
    )
    args = parser.parse_args()

    os.makedirs(DISP_DIR, exist_ok=True)

    keys_to_try = ['newmark', 'pinn']
    if args.source == 'newmark':
        keys_to_try = ['newmark']
    elif args.source == 'pinn':
        keys_to_try = ['pinn']

    for case in INPUT_VELOCITY_CASES:
        print(f'\n{"="*60}')
        print(f'  Case: {case}')
        print(f'{"="*60}')

        datasets  = {}
        v_in_case = None

        for src_key in keys_to_try:
            data = load_case(src_key, case)
            if data is None:
                print(f'  [{src_key}] not found — '
                      f'{SOURCE_DIRS[src_key]}/{src_key}_{case}.npz')
                continue
            label = SOURCE_LABELS[src_key]
            datasets[label] = data
            v_in_case = float(data['v_in'])
            print(f'  Loaded  {SOURCE_DIRS[src_key]}/{src_key}_{case}.npz'
                  f'  ({len(data["t"])} steps)')

        if not datasets:
            print(f'  No data found for case "{case}" — skipping.')
            continue

        print('\n  [1] Per-DOF energy spectrum (FFT)')
        plot_fft_per_dof(datasets, case, v_in_case, DISP_DIR)

        print('  [2] 2-D FFT dispersion map')
        plot_dispersion_2dfft(datasets, case, v_in_case, DISP_DIR)

    print(f'\nDone.  All figures saved to  {DISP_DIR}/')


if __name__ == '__main__':
    main()
