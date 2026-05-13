"""
dispersion.py
-------------
Post-processing script: loads saved responses from newmark.py and/or pinn.py
and produces two sets of plots for each case:

  1. Per-DOF FFT  — energy spectrum |FFT(x_flex_i)|² vs frequency (Hz).
                    x_flex_i = x_i - x_cm  removes the rigid-body drift that
                    dominates the raw FFT of a free-free chain with nonzero
                    total momentum.  Each DOF is a separate subplot.

  2. 2-D FFT dispersion map  — |FFT2(x_flex)| in the (κ, ω) plane with the
                               analytical dispersion relation overlaid.

Folder structure
----------------
  Results_Linear_Chain/
  ├── Newmark/      newmark_{case}.npz   (written by newmark.py)
  ├── PINN/         pinn_{case}.npz      (written by pinn.py)
  └── Dispersion/   all figures from this script

Usage
-----
    python dispersion.py                         # 20 DOFs, both sources
    python dispersion.py --ndof 10
    python dispersion.py --source newmark
    python dispersion.py --source pinn
    python dispersion.py --ndof 40 --window      # Hann window (off by default)
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

BASE_DIR    = 'Results_Linear_Chain'
NEWMARK_DIR = os.path.join(BASE_DIR, 'Newmark')
PINN_DIR    = os.path.join(BASE_DIR, 'PINN')
DISP_DIR    = os.path.join(BASE_DIR, 'Dispersion')

SOURCE_DIRS   = {'newmark': NEWMARK_DIR, 'pinn': PINN_DIR}
SOURCE_LABELS = {'newmark': 'Newmark-β', 'pinn': 'PINN'}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_case(source_key, case, n_dof):
    """
    Return dict of arrays from {source}_{n_dof}dof_{case}.npz, or None if missing.
    n_dof must match the value used when the solver was run.
    """
    path = os.path.join(SOURCE_DIRS[source_key],
                        f'{source_key}_{n_dof}dof_{case}.npz')
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ═══════════════════════════════════════════════════════════════════════════════
# Rigid-body removal
# ═══════════════════════════════════════════════════════════════════════════════

def remove_rigid_body(x):
    """
    Subtract the instantaneous center-of-mass displacement from every DOF.

    For a free-free chain the rigid-body mode is uniform translation:
        x_cm(t) = mean_i { x_i(t) }
        x_flex_i(t) = x_i(t) - x_cm(t)

    Parameters
    ----------
    x : (n_time, n_dof)

    Returns
    -------
    x_flex : (n_time, n_dof)  — rigid-body-free displacements
    x_cm   : (n_time,)        — center-of-mass trajectory
    """
    x_cm   = np.mean(x, axis=1)                      # (n_time,)
    x_flex = x - x_cm[:, np.newaxis]                 # broadcast over DOFs
    return x_flex, x_cm


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical helpers
# ═══════════════════════════════════════════════════════════════════════════════

def chain_eigenfreqs_hz(n, m=M_VAL, k=K_VAL):
    """
    Eigenfrequencies in Hz for a free-free uniform chain:
        f_j = (1/2π) · 2√(k/m) |sin(jπ/(2n))|   j = 0…n-1
    j=0 is the rigid-body mode (f=0, excluded from markers).
    """
    j = np.arange(n)
    omega_j = 2.0 * np.sqrt(k / m) * np.abs(np.sin(j * np.pi / (2.0 * n)))
    return omega_j / (2.0 * np.pi)


def analytical_dispersion(d=LATTICE_SPACING, m=M_VAL, k=K_VAL, n_pts=500):
    """ω(κ) = 2√(k/m)|sin(κd/2)|  →  (kappa, omega) arrays."""
    kappa = np.linspace(0.0, np.pi / d, n_pts)
    omega = 2.0 * np.sqrt(k / m) * np.abs(np.sin(kappa * d / 2.0))
    return kappa, omega


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — Per-DOF energy spectrum  (1-D FFT)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fft_per_dof(datasets, case, v_in, n_dof, out_dir, use_window=False):
    """
    One subplot per DOF in SELECTED_DOFS (auto-computed from n_dof).

    x-axis : frequency  f  (Hz)
    y-axis : energy spectrum  |FFT(x_flex_i)|²  (m²),  log scale

    Rigid-body drift is removed by subtracting x_cm(t) = mean_i{x_i(t)}
    before the FFT so that flexible modal peaks are clearly visible.

    Parameters
    ----------
    n_dof      : int  — number of DOFs (sets eigenfrequency markers and DOFs to plot)
    use_window : bool — apply a Hann window before FFT (default False)

    Saved as:  fft_energy_per_dof_{n_dof}dof_{case}.png
    """
    f_eig    = chain_eigenfreqs_hz(n=n_dof)
    f_max_hz = (2.0 * np.sqrt(K_VAL / M_VAL)) / (2.0 * np.pi) * 1.15

    # 5 evenly-spaced DOFs, always including first and last
    n_plot        = min(5, n_dof)
    selected_dofs = list(dict.fromkeys(
        np.linspace(0, n_dof - 1, n_plot, dtype=int).tolist()
    ))

    colors = ['k', 'C1', 'C0', 'C3']
    styles = ['-', '--', '-.', ':']

    n_sel = len(selected_dofs)
    fig, axes = plt.subplots(n_sel, 1,
                             figsize=(11, 3 * n_sel),
                             sharex=True)

    for ax, di in zip(axes, selected_dofs):
        for idx, (label, data) in enumerate(datasets.items()):
            t   = data['t']
            x   = data['x']                                    # (n_time, n_dof)
            dt  = float(t[1] - t[0])
            n_t = len(t)

            # ── remove rigid-body drift ────────────────────────────────────────
            x_flex, _ = remove_rigid_body(x)
            sig = x_flex[:, di]

            # ── optional Hann window ───────────────────────────────────────────
            if use_window:
                sig = sig * np.hanning(n_t)

            freqs  = np.fft.rfftfreq(n_t, d=dt)               # Hz
            energy = np.abs(np.fft.rfft(sig)) ** 2            # m²

            ax.plot(freqs, energy,
                    color=colors[idx % len(colors)],
                    ls=styles[idx % len(styles)],
                    lw=1.3, alpha=0.9, label=label)

        # mark flexible eigenfrequencies (skip j=0 rigid-body at f=0)
        for fe in f_eig[1:]:
            ax.axvline(fe, color='C2', lw=0.8, ls=':', alpha=0.65)

        ax.set_ylabel(f'DOF {di+1}\n|X(f)|²  (m²)', fontsize=9)
        ax.set_yscale('log')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3, which='both')

    axes[-1].set_xlabel('Frequency  f  (Hz)')
    axes[-1].set_xlim(0.0, f_max_hz)

    win_note = '  [Hann window]' if use_window else '  [no window]'
    label_str = '  vs  '.join(datasets.keys())
    fig.suptitle(
        f'Energy spectrum per DOF  [{label_str}]{win_note}\n'
        f'{n_dof} DOFs  |  {case.capitalize()}  v₀ = {v_in:.1f} m/s'
        f'   (rigid-body drift removed;  green dotted = eigenfrequencies)',
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f'fft_energy_per_dof_{n_dof}dof_{case}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — 2-D FFT dispersion map
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_2dfft(x_tn, dt, d=LATTICE_SPACING, use_window=False):
    """
    2-D FFT of a space-time displacement matrix.

    Rigid-body drift is removed first by subtracting the instantaneous
    center-of-mass from every DOF at every time step.  A temporal-mean
    subtraction per DOF is then applied to remove any residual DC.

    Parameters
    ----------
    x_tn      : (n_time, n_dof)
    dt        : temporal sampling interval (s)
    d         : lattice spacing (m)
    use_window: apply 2-D Hann window before FFT (default False)

    Returns
    -------
    k     : (n_k,)     κ ∈ [0, π/d]  (rad/m)
    omega : (n_ω,)     ω ≥ 0          (rad/s)
    S     : (n_ω, n_k) normalised |FFT2|
    """
    x_tn = np.asarray(x_tn, dtype=float)
    n_t, n_x = x_tn.shape

    # ── remove rigid-body drift (k=0 mode) ────────────────────────────────────
    x_flex, _ = remove_rigid_body(x_tn)

    # ── subtract residual temporal mean per DOF ────────────────────────────────
    x0 = x_flex - np.mean(x_flex, axis=0, keepdims=True)

    # ── optional 2-D Hann window ──────────────────────────────────────────────
    if use_window:
        wt = np.hanning(n_t)[:, None]
        wx = np.hanning(n_x)[None, :]
        x0 = x0 * wt * wx

    F_full = np.fft.fft2(x0)
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


def plot_dispersion_2dfft(datasets, case, v_in, n_dof, out_dir,
                          d=LATTICE_SPACING, use_window=False):
    """
    2-D FFT dispersion map — one panel per source, side by side.

    x-axis : normalised wavenumber  κ/(π/d) ∈ [0, 1]
    y-axis : angular frequency  ω  (rad/s)

    Analytical dispersion ω(κ)=2√(k/m)|sin(κd/2)| overlaid in white dashed.

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

        k, omega, S = _compute_2dfft(x, dt, d, use_window=use_window)
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
    win_note  = '  [Hann window]' if use_window else '  [no window]'
    label_str = '  vs  '.join(datasets.keys())
    fig.suptitle(
        f'2-D FFT dispersion map  [{label_str}]{win_note}\n'
        f'{n_dof} DOFs  |  {case.capitalize()}  v₀ = {v_in:.1f} m/s'
        f'   (rigid-body drift removed)',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f'dispersion_2dfft_{n_dof}dof_{case}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — Extracted dispersion curve  (peak-picking the 2-D FFT)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_peaks(k, omega, S, peak_threshold=0.05):
    """
    For every wavenumber bin, find the angular frequency of peak spectral energy.

    A column is skipped when its maximum is below  peak_threshold × global_max
    so that wavenumber bins with negligible energy are excluded from the curve.

    Parameters
    ----------
    k, omega       : 1-D arrays of wavenumber (rad/m) and frequency (rad/s)
    S              : (n_ω, n_k) normalised |FFT2|
    peak_threshold : fraction of global max below which a column is ignored

    Returns
    -------
    k_out : (m,)  wavenumber values where a valid peak exists
    w_out : (m,)  corresponding peak angular frequency
    """
    global_max = S.max()
    k_out, w_out = [], []
    for j in range(S.shape[1]):
        if S[:, j].max() < peak_threshold * global_max:
            continue
        i_peak = int(np.argmax(S[:, j]))
        k_out.append(k[j])
        w_out.append(omega[i_peak])
    return np.array(k_out), np.array(w_out)


def plot_dispersion_curve(datasets, case, v_in, n_dof, out_dir,
                          d=LATTICE_SPACING, use_window=False,
                          peak_threshold=0.05):
    """
    Extract the dispersion curve from the 2-D FFT by peak-picking ω at each
    κ bin, then compare directly against the analytical relation.

    One plot per case; all sources overlaid on the same axes.

    x-axis : normalised wavenumber  κ/(π/d) ∈ [0, 1]
    y-axis : angular frequency  ω  (rad/s)

    Analytical relation : solid black line
    Extracted points    : coloured markers (one colour per source)

    Saved as:  dispersion_curve_{n_dof}dof_{case}.png
    """
    kappa_ana, omega_ana = analytical_dispersion(d)
    kpi_ana   = kappa_ana / (np.pi / d)
    omega_max = 2.0 * np.sqrt(K_VAL / M_VAL)

    src_markers = ['o', 's', '^', 'D']
    src_colors  = ['C0', 'C1', 'C2', 'C3']

    fig, ax = plt.subplots(figsize=(7, 5))

    # analytical dispersion — reference line
    ax.plot(kpi_ana, omega_ana, 'k-', lw=2.2, label='Analytical  ω(κ)', zorder=5)

    for idx, (label, data) in enumerate(datasets.items()):
        t  = data['t']
        x  = data['x']
        dt = float(t[1] - t[0])

        k, omega, S = _compute_2dfft(x, dt, d, use_window=use_window)
        kpi = k / (np.pi / d)

        kpi_peaks, omega_peaks = _extract_peaks(kpi, omega, S, peak_threshold)

        ax.scatter(kpi_peaks, omega_peaks,
                   marker=src_markers[idx % len(src_markers)],
                   color=src_colors[idx  % len(src_colors)],
                   s=70, zorder=4, alpha=0.9,
                   label=f'{label}  (peak-picked)')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, omega_max * 1.2)
    ax.set_xlabel('κ / (π/d)  —  normalised wavenumber', fontsize=11)
    ax.set_ylabel('ω  (rad/s)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    win_note  = '  [Hann window]' if use_window else '  [no window]'
    label_str = '  vs  '.join(datasets.keys())
    ax.set_title(
        f'Dispersion curve  [{label_str}]{win_note}\n'
        f'{n_dof} DOFs  |  {case.capitalize()}  v₀ = {v_in:.1f} m/s',
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f'dispersion_curve_{n_dof}dof_{case}.png')
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
    parser.add_argument('--ndof', type=int, default=N_DOF,
                        help='Number of DOFs used when the solvers were run '
                             '(default: %(default)s)')
    parser.add_argument('--source', choices=['newmark', 'pinn', 'both'],
                        default='both',
                        help='Which solver results to load (default: both)')
    parser.add_argument('--window', action='store_true',
                        help='Apply Hann window before FFT (default: off)')
    parser.add_argument('--peak-threshold', type=float, default=0.05,
                        dest='peak_threshold',
                        help='Fraction of global FFT max below which a '
                             'wavenumber bin is excluded from peak-picking '
                             '(default: 0.05)')
    args = parser.parse_args()

    n_dof = args.ndof
    if n_dof < 2:
        parser.error('--ndof must be >= 2')

    os.makedirs(DISP_DIR, exist_ok=True)

    keys_to_try = ['newmark', 'pinn']
    if args.source == 'newmark':
        keys_to_try = ['newmark']
    elif args.source == 'pinn':
        keys_to_try = ['pinn']

    print(f'N_DOF = {n_dof}  |  Hann window: {"ON" if args.window else "OFF"}'
          f'  |  peak threshold: {args.peak_threshold}')

    for case in INPUT_VELOCITY_CASES:
        print(f'\n{"="*60}')
        print(f'  Case: {case}')
        print(f'{"="*60}')

        datasets  = {}
        v_in_case = None

        for src_key in keys_to_try:
            data = load_case(src_key, case, n_dof)
            if data is None:
                stem = f'{src_key}_{n_dof}dof_{case}.npz'
                print(f'  [{src_key}] not found — '
                      f'{SOURCE_DIRS[src_key]}/{stem}')
                continue
            label = SOURCE_LABELS[src_key]
            datasets[label] = data
            v_in_case = float(data['v_in'])
            print(f'  Loaded  {SOURCE_DIRS[src_key]}/{src_key}_{n_dof}dof_{case}.npz'
                  f'  ({len(data["t"])} steps,  n_dof={data["x"].shape[1]})')

        if not datasets:
            print(f'  No data found for case "{case}" — skipping.')
            continue

        print('\n  [1] Per-DOF energy spectrum (FFT, rigid-body removed)')
        plot_fft_per_dof(datasets, case, v_in_case, n_dof, DISP_DIR,
                         use_window=args.window)

        print('  [2] 2-D FFT dispersion map (rigid-body removed)')
        plot_dispersion_2dfft(datasets, case, v_in_case, n_dof, DISP_DIR,
                              use_window=args.window)

        print('  [3] Extracted dispersion curve vs analytical')
        plot_dispersion_curve(datasets, case, v_in_case, n_dof, DISP_DIR,
                              use_window=args.window,
                              peak_threshold=args.peak_threshold)

    print(f'\nDone.  All figures saved to  {DISP_DIR}/')


if __name__ == '__main__':
    main()
