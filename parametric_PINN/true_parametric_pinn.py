"""
true_parametric_pinn.py
=======================
True parametric PINN architecture with force parameters (phi1, phi2) as inputs.

Model mapping
-------------
    (t, phi1, phi2) -> x(t; phi1, phi2)

This allows one single model to interpolate inside the trained force-parameter
range, unlike case-by-case retraining.

Note
----
This module targets the continuous ODE stage by default. For long horizons with
many impacts, you can train in supervised mode using stitched non-parametric
PINN trajectories (train_supervised) and then predict unseen in-range cases.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class ParametricModelConfig:
    n_dof: int = 20
    m_x: float = 1.0
    k: float = 1.0
    c: float = 0.0
    seg_window: float = 1.0

    # Parametric domain bounds for normalisation
    phi1_min: float = 0.0
    phi1_max: float = 1.0
    phi2_min: float = 0.0
    phi2_max: float = 1.0

    # Training
    layers: Tuple[int, ...] = (3, 64, 64, 20)  # input: [t, phi1, phi2]
    beta_ic: float = 1.0
    beta_ode: float = 1.0
    beta_data: float = 1.0
    lr: float = 1e-3


class TrueParametricPINN:
    """Parametric PINN for x(t; phi1, phi2) under fixed structure settings."""

    def __init__(self, cfg: ParametricModelConfig):
        self.cfg = cfg
        self.n_dof = cfg.n_dof

        self.M_tf, self.K_tf, self.C_tf = self._build_system_matrices(cfg)

        self.lb = tf.constant(
            np.array([[0.0, cfg.phi1_min, cfg.phi2_min]], dtype=np.float32)
        )
        self.ub = tf.constant(
            np.array([[cfg.seg_window, cfg.phi1_max, cfg.phi2_max]], dtype=np.float32)
        )

        self.weights, self.biases = self._init_nn(list(cfg.layers))
        self.trainable_vars = self.weights + self.biases
        self.opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

        self.loss_log = []
        self.loss_mode = ""

    @staticmethod
    def _build_system_matrices(cfg: ParametricModelConfig):
        M = np.diag(cfg.m_x * np.ones(cfg.n_dof))
        K = np.zeros((cfg.n_dof, cfg.n_dof))
        for i in range(cfg.n_dof):
            K[i, i] = 2 * cfg.k if 0 < i < cfg.n_dof - 1 else cfg.k
            if i > 0:
                K[i, i - 1] = -cfg.k
            if i < cfg.n_dof - 1:
                K[i, i + 1] = -cfg.k
        C = (cfg.c / cfg.k) * K if cfg.k != 0 else np.zeros_like(K)
        return (
            tf.constant(M, dtype=tf.float32),
            tf.constant(K, dtype=tf.float32),
            tf.constant(C, dtype=tf.float32),
        )

    @staticmethod
    def _xavier(shape):
        std = np.sqrt(2.0 / (shape[0] + shape[1]))
        return tf.Variable(tf.random.truncated_normal(shape, stddev=std, dtype=tf.float32))

    def _init_nn(self, layers):
        w, b = [], []
        for i in range(len(layers) - 1):
            w.append(self._xavier([layers[i], layers[i + 1]]))
            b.append(tf.Variable(tf.zeros([1, layers[i + 1]], dtype=tf.float32)))
        return w, b

    def _nn(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb + 1e-12) - 1.0
        for W, B in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + B)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]

    def _net_u(self, X):
        """
        X: [N, 3] with columns [t, phi1, phi2]
        Returns x, x_t, x_tt where derivatives are wrt t only.
        """
        t = X[:, 0:1]
        phi = X[:, 1:3]

        with tf.GradientTape() as tape2:
            tape2.watch(t)
            with tf.GradientTape() as tape1:
                tape1.watch(t)
                X_in = tf.concat([t, phi], axis=1)
                x = self._nn(X_in)
            x_t = tf.squeeze(tape1.batch_jacobian(x, t), axis=-1)
        x_tt = tf.squeeze(tape2.batch_jacobian(x_t, t), axis=-1)

        return x, x_t, x_tt

    def _net_f(self, X):
        x, x_t, x_tt = self._net_u(X)

        t = X[:, 0:1]
        phi1 = X[:, 1:2]
        phi2 = X[:, 2:3]

        F_last = phi1 * tf.sin(phi2 * np.pi * t)
        F = tf.concat([tf.zeros_like(t)] * (self.n_dof - 1) + [F_last], axis=1)

        residual = (
            tf.transpose(tf.matmul(self.M_tf, x_tt, transpose_b=True))
            + tf.transpose(tf.matmul(self.C_tf, x_t, transpose_b=True))
            + tf.transpose(tf.matmul(self.K_tf, x, transpose_b=True))
            - F
        )
        return residual

    def _loss_physics(self, X_col, X_ic, x_ic, xt_ic):
        x0, xt0, _ = self._net_u(X_ic)
        loss_ic = tf.reduce_mean(tf.square(x0 - x_ic)) + tf.reduce_mean(tf.square(xt0 - xt_ic))
        loss_ode = tf.reduce_mean(tf.square(self._net_f(X_col)))
        total = self.cfg.beta_ic * loss_ic + self.cfg.beta_ode * loss_ode
        return total, loss_ic, loss_ode

    @tf.function
    def _train_step_physics(self, X_col, X_ic, x_ic, xt_ic):
        with tf.GradientTape() as tape:
            total, loss_ic, loss_ode = self._loss_physics(X_col, X_ic, x_ic, xt_ic)
        grads = tape.gradient(total, self.trainable_vars)
        self.opt.apply_gradients(zip(grads, self.trainable_vars))
        return total, loss_ic, loss_ode

    def train(self, X_col, X_ic, x_ic, xt_ic, n_iter=5000, print_every=500):
        """Physics-informed training (ODE + IC)."""
        X_col = tf.constant(X_col, dtype=tf.float32)
        X_ic = tf.constant(X_ic, dtype=tf.float32)
        x_ic = tf.constant(x_ic, dtype=tf.float32)
        xt_ic = tf.constant(xt_ic, dtype=tf.float32)

        self.loss_mode = "physics"
        self.loss_log = []

        for it in range(1, n_iter + 1):
            total, loss_ic, loss_ode = self._train_step_physics(X_col, X_ic, x_ic, xt_ic)
            if it % print_every == 0 or it == 1:
                print(
                    f"Iter {it:6d} | total={float(total):.3e} "
                    f"ic={float(loss_ic):.3e} ode={float(loss_ode):.3e}"
                )
            self.loss_log.append(float(total))

    @tf.function
    def _train_step_supervised(self, X_data, y_data):
        with tf.GradientTape() as tape:
            y_pred = self._nn(X_data)
            loss_data = tf.reduce_mean(tf.square(y_pred - y_data))
            total = self.cfg.beta_data * loss_data
        grads = tape.gradient(total, self.trainable_vars)
        self.opt.apply_gradients(zip(grads, self.trainable_vars))
        return total, loss_data

    def train_supervised(self, X_data, y_data, n_iter=5000, print_every=500):
        """
        Supervised training from stitched non-parametric trajectories.

        Use this mode when targets over full impact horizons are available.
        """
        X_data = tf.constant(X_data, dtype=tf.float32)
        y_data = tf.constant(y_data, dtype=tf.float32)

        self.loss_mode = "supervised"
        self.loss_log = []

        for it in range(1, n_iter + 1):
            total, loss_data = self._train_step_supervised(X_data, y_data)
            if it % print_every == 0 or it == 1:
                print(f"Iter {it:6d} | total={float(total):.3e} data={float(loss_data):.3e}")
            self.loss_log.append(float(total))

    def predict(self, t: np.ndarray, phi1: float, phi2: float):
        t = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        phi1_col = np.full_like(t, float(phi1), dtype=np.float32)
        phi2_col = np.full_like(t, float(phi2), dtype=np.float32)
        X = np.hstack([t, phi1_col, phi2_col]).astype(np.float32)
        x, xt, _ = self._net_u(tf.constant(X, dtype=tf.float32))
        return x.numpy(), xt.numpy()

    def save_checkpoint(self, out_dir: str | Path, tag: str = "true_parametric_pinn"):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save weights/biases
        np.savez(
            out / f"{tag}_weights.npz",
            **{f"W{i}": w.numpy() for i, w in enumerate(self.weights)},
            **{f"b{i}": b.numpy() for i, b in enumerate(self.biases)},
        )

        # Save config + training info
        meta = {
            "config": asdict(self.cfg),
            "loss_mode": self.loss_mode,
            "loss_log": self.loss_log,
        }
        with open(out / f"{tag}_training.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_checkpoint(self, out_dir: str | Path, tag: str = "true_parametric_pinn"):
        out = Path(out_dir)
        w_data = np.load(out / f"{tag}_weights.npz")

        for i, w in enumerate(self.weights):
            w.assign(w_data[f"W{i}"])
        for i, b in enumerate(self.biases):
            b.assign(w_data[f"b{i}"])

        with open(out / f"{tag}_training.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.loss_mode = meta.get("loss_mode", "")
        self.loss_log = list(meta.get("loss_log", []))


def build_parametric_training_data(
    phi_cases: Iterable[Tuple[float, float]],
    n_collocation_t: int,
    seg_window: float,
    n_dof: int,
):
    """
    Build collocation + IC data for true parametric PINN.

    - Collocation: (t, phi1, phi2) over all listed cases
    - IC: t=0 for each case, x=0, x_t=0
    """
    X_col_list, X_ic_list = [], []
    x_ic_list, xt_ic_list = [], []

    t_col = np.linspace(0, seg_window, n_collocation_t, dtype=np.float32).reshape(-1, 1)

    for p1, p2 in phi_cases:
        p1_col = np.full_like(t_col, float(p1), dtype=np.float32)
        p2_col = np.full_like(t_col, float(p2), dtype=np.float32)
        X_col_list.append(np.hstack([t_col, p1_col, p2_col]))

        X_ic_list.append(np.array([[0.0, float(p1), float(p2)]], dtype=np.float32))
        x_ic_list.append(np.zeros((1, n_dof), dtype=np.float32))
        xt_ic_list.append(np.zeros((1, n_dof), dtype=np.float32))

    X_col = np.vstack(X_col_list)
    X_ic = np.vstack(X_ic_list)
    x_ic = np.vstack(x_ic_list)
    xt_ic = np.vstack(xt_ic_list)

    return X_col, X_ic, x_ic, xt_ic


def build_supervised_dataset(
    t_list: Iterable[np.ndarray],
    x_list: Iterable[np.ndarray],
    phi_cases: Iterable[Tuple[float, float]],
):
    """
    Build supervised dataset from full trajectories.

    Inputs
    ------
    t_list: list of (Nt,) or (Nt,1)
    x_list: list of (Nt, n_dof)
    phi_cases: list of (phi1, phi2)

    Returns
    -------
    X_data: (sum Nt, 3) as [t, phi1, phi2]
    Y_data: (sum Nt, n_dof)
    """
    X_data_parts, Y_data_parts = [], []

    for t_arr, x_arr, (p1, p2) in zip(t_list, x_list, phi_cases):
        t = np.asarray(t_arr, dtype=np.float32).reshape(-1, 1)
        x = np.asarray(x_arr, dtype=np.float32)

        p1_col = np.full_like(t, float(p1), dtype=np.float32)
        p2_col = np.full_like(t, float(p2), dtype=np.float32)

        X_data_parts.append(np.hstack([t, p1_col, p2_col]))
        Y_data_parts.append(x)

    X_data = np.vstack(X_data_parts).astype(np.float32)
    Y_data = np.vstack(Y_data_parts).astype(np.float32)
    return X_data, Y_data
