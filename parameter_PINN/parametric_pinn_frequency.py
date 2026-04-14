"""Frequency-parametric PINN trial for the impact-damper chain.

This script reuses the original TensorFlow2 PINN implementation and sweeps
input excitation frequencies (Hz) as the first parametric study.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.io import savemat

# Reuse the original model implementation
ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "PINN"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from pinn_ndof_chain_tf2 import PIPNNs  # noqa: E402


def build_default_system(n_dof: int = 2):
    """Create a small default chain system for first-trial parametric runs."""
    M = np.eye(n_dof, dtype=np.float32)
    K = np.array([[2.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    C = 0.01 * np.eye(n_dof, dtype=np.float32)

    x0 = np.zeros((1, n_dof), dtype=np.float32)
    xt0 = np.zeros((1, n_dof), dtype=np.float32)
    y0 = np.zeros((1, n_dof), dtype=np.float32)
    yt0 = np.zeros((1, n_dof), dtype=np.float32)

    D = np.full((n_dof,), 0.05, dtype=np.float32)
    return M, K, C, x0, xt0, y0, yt0, D


def run_frequency_case(freq_hz: float, tmax: float, npts: int, adam_iter: int):
    n_dof = 2
    M, K, C, x0, xt0, y0, yt0, D = build_default_system(n_dof=n_dof)

    t = np.linspace(0.0, tmax, npts, dtype=np.float32).reshape(-1, 1)
    t0 = np.array([[0.0]], dtype=np.float32)
    lb = np.array([0.0], dtype=np.float32)
    ub = np.array([tmax], dtype=np.float32)

    model = PIPNNs(
        lb=lb,
        ub=ub,
        t0=t0,
        t=t,
        x0_total=x0,
        xt0_total=xt0,
        y0=y0,
        yt0=yt0,
        M=M,
        K=K,
        C=C,
        D=D,
        n_dof=n_dof,
        phi=0.0,
        phi1=1.0,
        phi2=2.0 * np.pi * freq_hz,
        layers=[1, 64, 64, n_dof],
        hyp_ini_weight_loss=[1.0, 1.0],
        optimizer_LB=False,
    )
    model.train(nIter=adam_iter, optimizer_LB=False, print_every=max(1, adam_iter // 5))
    x, xt, xtt = model.predict(t)
    return {
        "freq_hz": freq_hz,
        "time": t.squeeze(-1),
        "x": x,
        "xt": xt,
        "xtt": xtt,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Parametric PINN sweep by input frequency.")
    p.add_argument("--freqs", nargs="+", type=float, required=True, help="Frequencies in Hz.")
    p.add_argument("--tmax", type=float, default=1.0, help="Total simulation time.")
    p.add_argument("--npts", type=int, default=200, help="Number of collocation/prediction points.")
    p.add_argument("--adam_iter", type=int, default=200, help="Adam training iterations per frequency.")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "Result" / "parametric_pinn_data.mat",
        help="Output MAT file path.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_cases = []
    for f in args.freqs:
        print(f"Running frequency case: {f:.3f} Hz")
        all_cases.append(run_frequency_case(f, args.tmax, args.npts, args.adam_iter))

    freq_hz = np.array([c["freq_hz"] for c in all_cases], dtype=np.float32)
    time = all_cases[0]["time"]
    x_stack = np.stack([c["x"] for c in all_cases], axis=0)
    xt_stack = np.stack([c["xt"] for c in all_cases], axis=0)
    xtt_stack = np.stack([c["xtt"] for c in all_cases], axis=0)

    savemat(
        args.out,
        {
            "freq_hz": freq_hz,
            "time": time,
            "x": x_stack,
            "xt": xt_stack,
            "xtt": xtt_stack,
        },
    )
    print(f"Saved parametric results to: {args.out}")


if __name__ == "__main__":
    main()
