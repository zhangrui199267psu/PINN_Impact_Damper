"""
parametric_pinn_force_sweep.py
--------------------------------
Parametric PINN runner where ONLY forcing parameters vary:
  - phi1: forcing amplitude
  - phi2: forcing frequency (in phi1*sin(phi2*pi*(t+phi)))
All structural parameters remain fixed constants.

Outputs
-------
For each (phi1, phi2) case, saves:
  Result/pinn_data_phi1_<...>_phi2_<...>.mat
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PINN"))
sys.path.insert(0, str(ROOT / "Dispersion"))

from pinn_ndof_chain_tf2 import (  # noqa: E402
    PIPNNs,
    find_impact_times,
    propagate_ics,
    newmark_beta,
)
from pinn_dispersion_from_mat import save_pinn_results  # noqa: E402


@dataclass(frozen=True)
class StructuralConfig:
    n_dof: int = 20
    m_x: float = 1.0
    m_y: float = 0.3
    k: float = 1.0
    c: float = 0.0
    D: float = 1.0
    r: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    num_neurons: int = 64
    beta_icx: float = 1.0
    beta_fx: float = 1.0
    optimizer_lb: bool = True
    seg_window: float = 1.0
    n_points_per_seg: int = 1000
    n_iter_per_seg: int = 1000
    n_segments: int = 50


def build_system_matrices(cfg: StructuralConfig):
    M = np.diag(cfg.m_x * np.ones(cfg.n_dof))

    K = np.zeros((cfg.n_dof, cfg.n_dof))
    for i in range(cfg.n_dof):
        K[i, i] = 2 * cfg.k if 0 < i < cfg.n_dof - 1 else cfg.k
        if i > 0:
            K[i, i - 1] = -cfg.k
        if i < cfg.n_dof - 1:
            K[i, i + 1] = -cfg.k

    C = (cfg.c / cfg.k) * K if cfg.k != 0 else np.zeros_like(K)

    A = np.array([[cfg.m_x, cfg.m_y], [1.0, -1.0]])
    B = np.array([[cfg.m_x, cfg.m_y], [-cfg.r, cfg.r]])
    A_inv_B = np.linalg.inv(A) @ B
    return M, K, C, A_inv_B


def simulate_case(phi1: float, phi2: float, s_cfg: StructuralConfig, t_cfg: TrainingConfig):
    M_total, K_total, C_total, A_inv_B = build_system_matrices(s_cfg)

    lb = np.array([0.0])
    ub = np.array([t_cfg.seg_window])
    t0 = np.array([[0.0]])

    layers = [1, t_cfg.num_neurons, s_cfg.n_dof]
    hyp_ini_weight_loss = np.array([t_cfg.beta_icx, t_cfg.beta_fx])

    cur_x0 = np.zeros((1, s_cfg.n_dof))
    cur_xt0 = np.zeros((1, s_cfg.n_dof))
    cur_y0 = np.zeros(s_cfg.n_dof)
    cur_yt0 = np.zeros(s_cfg.n_dof)
    cur_phi = np.array([[0.0]])

    all_t_sim, all_x_sim, all_xt_sim, all_y_sim, all_t_imp = [], [], [], [], []
    t_cumulative = 0.0

    for seg in range(1, t_cfg.n_segments + 1):
        print(f"[phi1={phi1}, phi2={phi2}] Segment {seg}/{t_cfg.n_segments}")

        t_seg_arr = np.linspace(0, t_cfg.seg_window, t_cfg.n_points_per_seg).reshape(-1, 1)

        model = PIPNNs(
            lb,
            ub,
            t0,
            t_seg_arr,
            cur_x0,
            cur_xt0,
            cur_y0,
            cur_yt0,
            M_total,
            K_total,
            s_cfg.D,
            s_cfg.n_dof,
            cur_phi,
            phi1,
            phi2,
            layers,
            hyp_ini_weight_loss,
            C=C_total,
            optimizer_LB=t_cfg.optimizer_lb,
        )
        model.train(nIter=t_cfg.n_iter_per_seg, optimizer_LB=t_cfg.optimizer_lb)

        t_impacts, _ = find_impact_times(model, cur_y0, cur_yt0, s_cfg.D, t_cfg.seg_window)
        first_dof = int(np.argmin(t_impacts))
        t_imp = float(t_impacts[first_dof])

        t_sim = np.linspace(0, t_imp, t_cfg.n_points_per_seg + 1).reshape(-1, 1)
        x_sim, xt_sim, _ = model.predict(t_sim)

        y_sim_seg = (cur_y0 + cur_yt0 * t_sim).T  # (n_dof, Nt)

        all_t_sim.append(t_sim.flatten() + t_cumulative)
        all_x_sim.append(x_sim)
        all_xt_sim.append(xt_sim)
        all_y_sim.append(y_sim_seg.T)
        all_t_imp.append(t_impacts)

        x1_s, xt1_s, _ = model.predict(np.array([[t_imp]]))
        next_x0, next_xt0, next_y0, next_yt0 = propagate_ics(
            model,
            t_imp,
            x1_s,
            xt1_s,
            cur_y0,
            cur_yt0,
            first_dof,
            s_cfg.m_x * np.ones(s_cfg.n_dof),
            s_cfg.m_y * np.ones(s_cfg.n_dof),
            s_cfg.r,
            A_inv_B,
        )

        t_cumulative += t_imp
        cur_x0, cur_xt0, cur_y0, cur_yt0 = next_x0, next_xt0, next_y0, next_yt0
        cur_phi = cur_phi + t_imp

    t_total = np.concatenate(all_t_sim)
    x_PINN_total = np.vstack(all_x_sim)

    return t_total, x_PINN_total


def run_parametric_sweep(
    phi_cases: Iterable[Tuple[float, float]],
    s_cfg: StructuralConfig,
    t_cfg: TrainingConfig,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for phi1, phi2 in phi_cases:
        t_total, x_PINN_total = simulate_case(phi1, phi2, s_cfg, t_cfg)

        tag = f"phi1_{phi1:g}_phi2_{phi2:g}".replace(".", "p")
        out_path = out_dir / f"pinn_data_{tag}.mat"

        save_pinn_results(
            str(out_path),
            t_total,
            x_PINN_total,
            params={
                "n_dof": s_cfg.n_dof,
                "mx": s_cfg.m_x,
                "my": s_cfg.m_y,
                "k": s_cfg.k,
                "c": s_cfg.c,
                "D": s_cfg.D,
                "r": s_cfg.r,
                "phi1": phi1,
                "phi2": phi2,
            },
        )


if __name__ == "__main__":
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    structural_cfg = StructuralConfig()
    training_cfg = TrainingConfig()

    # Change ONLY phi1 and phi2 here.
    phi_param_cases = [
        (0.1, 0.5),
        (0.1, 1.0),
        (0.1, 2.0),
        (0.3, 1.0),
    ]

    run_parametric_sweep(
        phi_param_cases,
        structural_cfg,
        training_cfg,
        out_dir=ROOT / "Result",
    )

    print("\nParametric sweep complete.")
