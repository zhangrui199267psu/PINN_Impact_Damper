"""
continuity_multi_ridge_tracker.py
=================================
Continuity-based multi-ridge tracking on S(k, ω) from PINN .mat outputs.

Why this file
-------------
This script tracks ridges as 2D continuous objects in (k, ω), rather than
splitting only by frequency windows. It is designed to be more robust when
branches are noisy, broadened, or close to each other.

Pipeline
--------
1) Load .mat from PINN stage
2) Resample to uniform time grid
3) Compute one-sided S(k, ω) via 2D FFT
4) Build candidate peaks per k-row
5) Dynamic-programming continuity tracker (top-N ridges)
6) Plot heatmap + tracked ridges
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from pinn_dispersion_from_mat import (
    load_pinn_results,
    resample_to_uniform,
    compute_spectrum,
    linear_dispersion,
)


@dataclass
class TrackerConfig:
    n_ridges: int = 3
    omega_min: float = 0.01
    skip_transient: float = 0.20
    n_harmonic: int = 4
    pts_per_cycle: int = 12

    # Candidate peak extraction
    max_candidates_per_k: int = 6
    peak_prominence_db: float = 2.0

    # Continuity / scoring
    jump_penalty: float = 1.5
    amplitude_weight: float = 1.0

    # Ridge separation (suppression around tracked ridge)
    suppression_half_width_bins: int = 2

    # Plotting
    dB_range: float = 45.0
    omega_max_plot: float | None = None


def _row_candidates(power_row: np.ndarray, prominence_db: float, max_candidates: int) -> np.ndarray:
    """Return candidate ω-indices for one k-row, ranked by power descending."""
    row_db = 10.0 * np.log10(power_row + 1e-30)
    peaks, _ = find_peaks(row_db, prominence=prominence_db)

    if peaks.size == 0:
        peaks = np.array([int(np.argmax(power_row))], dtype=int)

    order = np.argsort(power_row[peaks])[::-1]
    peaks = peaks[order[:max_candidates]]
    return np.asarray(peaks, dtype=int)


def build_candidates(spectrum: np.ndarray, cfg: TrackerConfig) -> List[np.ndarray]:
    """Build candidate ω-indices per k-row from spectrum[k, ω]."""
    candidates = []
    for i_k in range(spectrum.shape[0]):
        candidates.append(
            _row_candidates(
                spectrum[i_k, :],
                prominence_db=cfg.peak_prominence_db,
                max_candidates=cfg.max_candidates_per_k,
            )
        )
    return candidates


def track_single_ridge_dp(
    omega_pos: np.ndarray,
    spectrum: np.ndarray,
    candidates_per_k: List[np.ndarray],
    jump_penalty: float,
    amplitude_weight: float,
) -> np.ndarray:
    """
    Track one continuous ridge across k using dynamic programming.

    Score to maximize:
        amplitude_weight * log(power) - jump_penalty * |Δω|_norm
    """
    n_k = spectrum.shape[0]
    omega_span = max(float(omega_pos[-1] - omega_pos[0]), 1e-12)

    # DP tables (ragged per k due to varying candidate counts)
    dp_vals: List[np.ndarray] = []
    back_ptrs: List[np.ndarray] = []

    # k=0 init
    c0 = candidates_per_k[0]
    s0 = amplitude_weight * np.log(spectrum[0, c0] + 1e-30)
    dp_vals.append(s0)
    back_ptrs.append(-np.ones_like(c0, dtype=int))

    # forward pass
    for k in range(1, n_k):
        c_prev = candidates_per_k[k - 1]
        c_cur = candidates_per_k[k]

        prev_scores = dp_vals[-1]
        cur_scores = np.full(c_cur.shape[0], -np.inf, dtype=float)
        cur_back = np.full(c_cur.shape[0], -1, dtype=int)

        for j, idx_cur in enumerate(c_cur):
            om_cur = omega_pos[idx_cur]
            base = amplitude_weight * np.log(spectrum[k, idx_cur] + 1e-30)

            # transition from each previous candidate
            d_om = np.abs(omega_pos[c_prev] - om_cur) / omega_span
            trans = prev_scores - jump_penalty * d_om + base

            i_best = int(np.argmax(trans))
            cur_scores[j] = trans[i_best]
            cur_back[j] = i_best

        dp_vals.append(cur_scores)
        back_ptrs.append(cur_back)

    # backtrack best endpoint at k=n_k-1
    last_idx = int(np.argmax(dp_vals[-1]))
    ridge_idx = np.full(n_k, -1, dtype=int)
    ridge_idx[-1] = int(candidates_per_k[-1][last_idx])

    cur = last_idx
    for k in range(n_k - 1, 0, -1):
        cur = int(back_ptrs[k][cur])
        ridge_idx[k - 1] = int(candidates_per_k[k - 1][cur])

    return ridge_idx


def suppress_around_ridge(spectrum: np.ndarray, ridge_idx: np.ndarray, half_width: int) -> np.ndarray:
    """Suppress power around an extracted ridge to reveal the next ridge."""
    out = spectrum.copy()
    n_k, n_om = out.shape
    for k in range(n_k):
        i0 = max(0, ridge_idx[k] - half_width)
        i1 = min(n_om, ridge_idx[k] + half_width + 1)
        out[k, i0:i1] *= 1e-8
    return out


def track_multi_ridges(
    omega_pos: np.ndarray,
    spectrum: np.ndarray,
    cfg: TrackerConfig,
) -> List[np.ndarray]:
    """Extract top-N continuous ridges iteratively with suppression."""
    work = spectrum.copy()
    ridges: List[np.ndarray] = []

    for _ in range(cfg.n_ridges):
        candidates = build_candidates(work, cfg)
        ridge_idx = track_single_ridge_dp(
            omega_pos=omega_pos,
            spectrum=work,
            candidates_per_k=candidates,
            jump_penalty=cfg.jump_penalty,
            amplitude_weight=cfg.amplitude_weight,
        )
        ridges.append(ridge_idx)
        work = suppress_around_ridge(work, ridge_idx, cfg.suppression_half_width_bins)

    return ridges


def run_tracker_from_mat(
    mat_path: str | Path,
    k_coupling: float,
    mx: float,
    cfg: TrackerConfig | None = None,
):
    """
    Compute S(k, ω), track continuous ridges, and return plotting arrays.
    """
    if cfg is None:
        cfg = TrackerConfig()

    t_total, x_PINN_total, params = load_pinn_results(str(mat_path))
    X_raw = x_PINN_total.T  # (N_time, n_dof) -> (n_dof, M)

    omega_max_lin = linear_dispersion(np.pi, k_coupling, mx)
    t_u, X_u, dt, omega_nyq = resample_to_uniform(
        t_total,
        X_raw,
        omega_max_lin,
        n_harmonic=cfg.n_harmonic,
        pts_per_cycle=cfg.pts_per_cycle,
    )

    k_pos, omega_pos, spectrum = compute_spectrum(
        t_u,
        X_u,
        skip_transient=cfg.skip_transient,
    )

    i0 = int(np.searchsorted(omega_pos, cfg.omega_min))
    omega_clip = omega_pos[i0:]
    spec_clip = spectrum[:, i0:]

    ridges_idx = track_multi_ridges(omega_clip, spec_clip, cfg)
    ridges_omega = [omega_clip[idx] for idx in ridges_idx]

    return {
        "k_pos": k_pos,
        "omega_pos": omega_clip,
        "spectrum": spec_clip,
        "ridges_omega": ridges_omega,
        "params": params,
        "omega_nyquist": omega_nyq,
    }


def plot_tracked_ridges(result: dict, cfg: TrackerConfig | None = None):
    """Plot spectrum heatmap with continuity-tracked ridges overlaid."""
    if cfg is None:
        cfg = TrackerConfig()

    k_pos = result["k_pos"]
    omega_pos = result["omega_pos"]
    spectrum = result["spectrum"]
    ridges_omega = result["ridges_omega"]

    S_dB = 10.0 * np.log10(spectrum + 1e-30)
    S_dB -= S_dB.max()
    S_dB = np.clip(S_dB, -cfg.dB_range, 0.0)

    omega_max = cfg.omega_max_plot if cfg.omega_max_plot is not None else omega_pos[-1]
    i_om = int(np.searchsorted(omega_pos, omega_max))
    i_om = max(i_om, 2)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(
        k_pos / np.pi,
        omega_pos[:i_om],
        S_dB[:, :i_om].T,
        shading="auto",
        cmap="inferno",
        vmin=-cfg.dB_range,
        vmax=0.0,
    )
    fig.colorbar(im, ax=ax, pad=0.02, label="Spectral power (dB)")

    for i, om in enumerate(ridges_omega, start=1):
        valid = np.isfinite(om)
        ax.plot(k_pos[valid] / np.pi, om[valid], lw=2.0, label=f"Tracked ridge {i}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, omega_pos[min(i_om - 1, len(omega_pos) - 1)])
    ax.set_xlabel(r"Wavenumber $k/\pi$")
    ax.set_ylabel(r"Frequency $\omega$ (rad/s)")
    ax.set_title("Continuity-based Multi-Ridge Tracking")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage (edit path/parameters):
    cfg = TrackerConfig(
        n_ridges=3,
        omega_min=0.01,
        skip_transient=0.20,
        n_harmonic=4,
        pts_per_cycle=12,
        jump_penalty=1.5,
        peak_prominence_db=2.0,
        suppression_half_width_bins=2,
    )

    mat_file = Path(__file__).resolve().parents[1] / "Result" / "pinn_data.mat"
    out = run_tracker_from_mat(mat_file, k_coupling=1.0, mx=1.0, cfg=cfg)
    plot_tracked_ridges(out, cfg=cfg)
