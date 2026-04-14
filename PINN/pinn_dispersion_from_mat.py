"""Post-process PINN MAT output and generate dispersion-related plots.

Usage:
    python PINN/pinn_dispersion_from_mat.py --mat Result/pinn_data.mat --out_dir Result/dispersion_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def pick_response_array(mat: dict) -> tuple[str, np.ndarray]:
    preferred = ["x", "disp", "u", "response", "X", "U"]
    for key in preferred:
        if key in mat:
            arr = np.array(mat[key])
            if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 2:
                return key, arr

    candidates: list[tuple[str, np.ndarray]] = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        arr = np.array(v)
        if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 2:
            candidates.append((k, arr))

    if not candidates:
        raise ValueError("No numeric 2D+ response array found in MAT file.")

    return max(candidates, key=lambda kv: kv[1].shape[0] * kv[1].shape[1])


def ensure_time_space(arr: np.ndarray) -> np.ndarray:
    x = np.array(arr)
    if x.ndim > 2:
        x = np.reshape(x, (x.shape[0], -1))
    if x.shape[0] < x.shape[1]:
        x = x.T
    return x - x.mean(axis=0, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate dispersion plots from pinn_data.mat")
    p.add_argument("--mat", type=Path, default=Path("Result/pinn_data.mat"), help="Path to MAT file")
    p.add_argument("--out_dir", type=Path, default=Path("Result/dispersion_plots"), help="Output folder for PNG plots")
    p.add_argument("--no_show", action="store_true", help="Disable interactive plot display")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mat = loadmat(args.mat)
    key, x_raw = pick_response_array(mat)
    x = ensure_time_space(x_raw)

    nt, nx = x.shape
    dt = float(np.squeeze(mat["dt"])) if "dt" in mat else 1.0
    dx = float(np.squeeze(mat["dx"])) if "dx" in mat else 1.0
    time = np.arange(nt) * dt
    space = np.arange(nx) * dx

    print(f"Using key: {key}, response shape: {x.shape}, dt={dt}, dx={dx}")

    fig = plt.figure(figsize=(8, 4.5))
    plt.pcolormesh(space, time, x, shading="auto", cmap="RdBu_r")
    plt.xlabel("Space index / coordinate")
    plt.ylabel("Time")
    plt.title("PINN response field")
    plt.colorbar(label="Displacement")
    plt.tight_layout()
    fig.savefig(args.out_dir / "01_time_space_field.png", dpi=180)

    fig = plt.figure(figsize=(8, 4))
    idxs = np.linspace(0, nx - 1, min(4, nx), dtype=int)
    for j in idxs:
        plt.plot(time, x[:, j], label=f"space #{j}")
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.title("Time traces at selected space points")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.out_dir / "02_time_traces.png", dpi=180)

    s = np.fft.fftshift(np.fft.fft2(x))
    pwr = np.abs(s) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    p_db = 10 * np.log10(pwr / (pwr.max() + 1e-12) + 1e-12)

    fig = plt.figure(figsize=(8, 5))
    plt.pcolormesh(k, f, p_db, shading="auto", cmap="magma")
    plt.xlabel("Wavenumber k [rad/unit]")
    plt.ylabel("Frequency f [1/unit]")
    plt.title("Dispersion-style f-k map")
    plt.colorbar(label="Power (dB, normalized)")
    plt.tight_layout()
    fig.savefig(args.out_dir / "03_fk_map.png", dpi=180)

    mask_f = f >= 0
    f_pos = f[mask_f]
    p_pos = pwr[mask_f, :]
    k_ridge = k[np.argmax(p_pos, axis=1)]

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(k_ridge, f_pos, ".", ms=3)
    plt.xlabel("Dominant wavenumber k")
    plt.ylabel("Frequency f")
    plt.title("Extracted dispersion ridge")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.out_dir / "04_dispersion_ridge.png", dpi=180)

    print(f"Saved plots to: {args.out_dir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
