"""
Parametric surrogate for free-free impact chain (velocity-parameterized).

Goal
----
Train one model with input (t, v0) so you can rapidly predict
- response x_i(t; v0),
- impact times for a queried initial velocity v0 inside the training range.

Notes
-----
1) This file is intentionally self-contained (no import from old PINN scripts).
2) Because impact events are discontinuous and their count changes with v0,
   we use a hybrid approach:
   - event-based simulator to generate training data,
   - parametric NN surrogate for response x(t, v0),
   - small regressor for first-N impact times vs v0.
3) It is "parametric PINN-style" in the sense of a parameterized neural surrogate,
   with optional physics-regularization hooks left in place for extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------

def build_free_free_chain_matrices(n_dof: int = 20, m_x: float = 1.0, k: float = 1.0, c: float = 0.0):
    """Build M, C, K for a free-free nearest-neighbor chain."""
    n = int(n_dof)
    M = m_x * np.eye(n)
    K = np.zeros((n, n), dtype=float)
    C = np.zeros((n, n), dtype=float)

    for i in range(n):
        if i > 0:
            K[i, i] += k
            K[i, i - 1] -= k
            C[i, i] += c
            C[i, i - 1] -= c
        if i < n - 1:
            K[i, i] += k
            K[i, i + 1] -= k
            C[i, i] += c
            C[i, i + 1] -= c
    return M, C, K


def impact_velocity_update(mx: float, my: float, r: float, xt_minus: float, yt_minus: float):
    """1D momentum + restitution update."""
    total = mx + my
    xt_plus = ((mx - r * my) * xt_minus + my * (1.0 + r) * yt_minus) / total
    yt_plus = (mx * (1.0 + r) * xt_minus + (my - r * mx) * yt_minus) / total
    return xt_plus, yt_plus


@dataclass
class SimulationParams:
    n_dof: int = 20
    m_x: float = 1.0
    m_y: float = 0.3
    k: float = 1.0
    c: float = 0.0
    D: float = 1.0
    r: float = 1.0
    t_end: float = 20.0
    dt: float = 1e-3


def simulate_event_chain(v0: float, p: SimulationParams):
    """
    Event-based time stepping for free-free chain with local impactors.

    Returns
    -------
    t: (nt,)
    x: (nt, n_dof)
    v: (nt, n_dof)
    y: (nt, n_dof)
    yt: (nt, n_dof)
    impacts: list of dict with {time, dof, impulse_abs}
    """
    n = p.n_dof
    nt = int(np.floor(p.t_end / p.dt)) + 1
    t = np.linspace(0.0, p.t_end, nt)

    M, C, K = build_free_free_chain_matrices(n, p.m_x, p.k, p.c)
    M_inv = np.linalg.inv(M)

    x = np.zeros((nt, n), dtype=float)
    v = np.zeros((nt, n), dtype=float)
    y = np.zeros((nt, n), dtype=float)
    yt = np.zeros((nt, n), dtype=float)

    # initial excitation only at left DOF
    v[0, 0] = float(v0)

    impacts: List[Dict[str, float]] = []

    for k in range(nt - 1):
        # free evolution for primary chain
        a = M_inv @ (-C @ v[k] - K @ x[k])
        v_trial = v[k] + p.dt * a
        x_trial = x[k] + p.dt * v_trial  # symplectic Euler

        # free flight of impactors
        y_trial = y[k] + p.dt * yt[k]
        yt_trial = yt[k].copy()

        # process impacts dof-by-dof at this step
        for i in range(n):
            rel_prev = abs(x[k, i] - y[k, i]) - p.D
            rel_now = abs(x_trial[i] - y_trial[i]) - p.D

            # trigger when crossing into contact from below
            if (rel_prev < 0.0) and (rel_now >= 0.0):
                xt_minus = v_trial[i]
                yt_minus = yt_trial[i]
                xt_plus, yt_plus = impact_velocity_update(p.m_x, p.m_y, p.r, xt_minus, yt_minus)
                v_trial[i] = xt_plus
                yt_trial[i] = yt_plus

                J = abs(p.m_x * (xt_plus - xt_minus))
                impacts.append({"time": float(t[k + 1]), "dof": int(i + 1), "impulse_abs": float(J)})

        x[k + 1] = x_trial
        v[k + 1] = v_trial
        y[k + 1] = y_trial
        yt[k + 1] = yt_trial

    return t, x, v, y, yt, impacts


# -----------------------------------------------------------------------------
# Parametric NN surrogates
# -----------------------------------------------------------------------------

def make_response_model(n_dof: int, width: int = 128, depth: int = 4) -> tf.keras.Model:
    """NN: (t, v0) -> x(t, v0) for all DOFs."""
    inp = tf.keras.Input(shape=(2,), name="tv")
    z = inp
    for _ in range(depth):
        z = tf.keras.layers.Dense(width, activation="tanh")(z)
    out = tf.keras.layers.Dense(n_dof, activation=None, name="x")(z)
    return tf.keras.Model(inp, out, name="param_response_surrogate")


def make_impact_time_model(n_impacts_max: int, width: int = 64, depth: int = 3) -> tf.keras.Model:
    """NN: v0 -> first N impact times (with mask during training)."""
    inp = tf.keras.Input(shape=(1,), name="v0")
    z = inp
    for _ in range(depth):
        z = tf.keras.layers.Dense(width, activation="tanh")(z)
    out = tf.keras.layers.Dense(n_impacts_max, activation=None, name="t_impacts")
    return tf.keras.Model(inp, out, name="param_impact_time_surrogate")


class ParametricImpactPINN:
    """
    One model for many initial velocities.

    Workflow:
    1) build_dataset(v0_grid) using event simulation,
    2) fit_response_surrogate(),
    3) fit_impact_time_surrogate(),
    4) predict_response(v0), predict_impact_times(v0).
    """

    def __init__(self, params: Optional[SimulationParams] = None):
        self.params = params or SimulationParams()
        self.response_model: Optional[tf.keras.Model] = None
        self.impact_model: Optional[tf.keras.Model] = None
        self.n_impacts_max: int = 0

        self._dataset_cache = None

    def build_dataset(self, v0_values: np.ndarray):
        p = self.params
        v0_values = np.asarray(v0_values, dtype=float).reshape(-1)

        X_rows = []
        Y_rows = []
        cases = {}

        # gather impact-time table (ragged -> padded)
        impact_time_lists: List[List[float]] = []

        for v0 in v0_values:
            t, x, v, y, yt, impacts = simulate_event_chain(float(v0), p)
            cases[float(v0)] = {"t": t, "x": x, "v": v, "y": y, "yt": yt, "impacts": impacts}

            tt = t.reshape(-1, 1)
            vv = np.full_like(tt, float(v0))
            X_tv = np.hstack([tt, vv])

            X_rows.append(X_tv)
            Y_rows.append(x)

            impact_time_lists.append([ev["time"] for ev in impacts])

        X = np.vstack(X_rows).astype(np.float32)
        Y = np.vstack(Y_rows).astype(np.float32)

        self.n_impacts_max = max((len(vv) for vv in impact_time_lists), default=0)
        if self.n_impacts_max == 0:
            T_imp = np.zeros((len(v0_values), 1), dtype=np.float32)
            M_imp = np.zeros_like(T_imp)
            self.n_impacts_max = 1
        else:
            T_imp = np.zeros((len(v0_values), self.n_impacts_max), dtype=np.float32)
            M_imp = np.zeros_like(T_imp)
            for i, lst in enumerate(impact_time_lists):
                n = len(lst)
                if n > 0:
                    T_imp[i, :n] = np.asarray(lst, dtype=np.float32)
                    M_imp[i, :n] = 1.0

        self._dataset_cache = {
            "v0_values": v0_values.astype(np.float32).reshape(-1, 1),
            "X_tv": X,
            "Y_x": Y,
            "T_imp": T_imp,
            "M_imp": M_imp,
            "cases": cases,
        }
        return self._dataset_cache

    def fit_response_surrogate(self, epochs: int = 200, batch_size: int = 4096, lr: float = 1e-3, verbose: int = 1):
        if self._dataset_cache is None:
            raise RuntimeError("Call build_dataset(...) first.")

        p = self.params
        X = self._dataset_cache["X_tv"]
        Y = self._dataset_cache["Y_x"]

        # normalize inputs
        t_scale = max(float(p.t_end), 1e-8)
        v_abs = np.max(np.abs(self._dataset_cache["v0_values"]))
        v_scale = max(float(v_abs), 1e-8)
        Xn = X.copy()
        Xn[:, 0] = Xn[:, 0] / t_scale
        Xn[:, 1] = Xn[:, 1] / v_scale

        self.response_model = make_response_model(n_dof=p.n_dof)
        self.response_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
        hist = self.response_model.fit(Xn, Y, epochs=epochs, batch_size=batch_size, verbose=verbose)

        self._dataset_cache["norm"] = {"t_scale": t_scale, "v_scale": v_scale}
        return hist

    def fit_impact_time_surrogate(self, epochs: int = 800, lr: float = 2e-3, verbose: int = 1):
        if self._dataset_cache is None:
            raise RuntimeError("Call build_dataset(...) first.")

        v0 = self._dataset_cache["v0_values"]
        T_imp = self._dataset_cache["T_imp"]
        M_imp = self._dataset_cache["M_imp"]

        v_scale = max(float(np.max(np.abs(v0))), 1e-8)
        vn = (v0 / v_scale).astype(np.float32)

        self.impact_model = make_impact_time_model(self.n_impacts_max)
        opt = tf.keras.optimizers.Adam(lr)

        x_tf = tf.constant(vn, dtype=tf.float32)
        y_tf = tf.constant(T_imp, dtype=tf.float32)
        m_tf = tf.constant(M_imp, dtype=tf.float32)

        for ep in range(epochs):
            with tf.GradientTape() as tape:
                pred = self.impact_model(x_tf, training=True)
                err2 = tf.square(pred - y_tf) * m_tf
                loss = tf.reduce_sum(err2) / (tf.reduce_sum(m_tf) + 1e-8)
            grads = tape.gradient(loss, self.impact_model.trainable_variables)
            opt.apply_gradients(zip(grads, self.impact_model.trainable_variables))
            if verbose and ((ep + 1) % 200 == 0 or ep == 0):
                print(f"[impact-model] epoch {ep+1}/{epochs}, loss={float(loss):.6e}")

        self._dataset_cache.setdefault("norm", {})["v_scale_imp"] = v_scale

    def predict_response(self, v0: float, t_query: np.ndarray) -> np.ndarray:
        if self.response_model is None:
            raise RuntimeError("fit_response_surrogate(...) has not been run.")
        norm = self._dataset_cache["norm"]
        t_query = np.asarray(t_query, dtype=float).reshape(-1, 1)
        x_in = np.hstack([t_query / norm["t_scale"], np.full_like(t_query, v0 / norm["v_scale"])])
        x_pred = self.response_model.predict(x_in.astype(np.float32), verbose=0)
        return x_pred

    def predict_impact_times(self, v0: float) -> np.ndarray:
        if self.impact_model is None:
            raise RuntimeError("fit_impact_time_surrogate(...) has not been run.")
        norm = self._dataset_cache.get("norm", {})
        v_scale = norm.get("v_scale_imp", 1.0)
        pred = self.impact_model.predict(np.array([[v0 / v_scale]], dtype=np.float32), verbose=0)[0]
        # enforce sorted nonnegative times
        pred = np.maximum(pred, 0.0)
        pred = np.sort(pred)
        return pred

    def save(self, folder: str):
        import os
        os.makedirs(folder, exist_ok=True)
        if self.response_model is not None:
            self.response_model.save(os.path.join(folder, "response_model.keras"))
        if self.impact_model is not None:
            self.impact_model.save(os.path.join(folder, "impact_time_model.keras"))
        if self._dataset_cache is not None:
            np.savez(
                os.path.join(folder, "meta.npz"),
                v0_values=self._dataset_cache["v0_values"],
                T_imp=self._dataset_cache["T_imp"],
                M_imp=self._dataset_cache["M_imp"],
            )


if __name__ == "__main__":
    # Minimal demo
    np.random.seed(1234)
    tf.random.set_seed(1234)

    params = SimulationParams(t_end=20.0, dt=1e-3, n_dof=20)
    model = ParametricImpactPINN(params)

    # training velocities (you can densify this grid)
    v0_train = np.array([-10.0, -6.0, -3.0, -2.0, -1.0], dtype=float)

    print("Building training dataset ...")
    model.build_dataset(v0_train)

    print("Training response surrogate ...")
    model.fit_response_surrogate(epochs=100, batch_size=8192, lr=1e-3, verbose=1)

    print("Training impact-time surrogate ...")
    model.fit_impact_time_surrogate(epochs=400, lr=2e-3, verbose=1)

    v0_test = -4.0
    tq = np.linspace(0.0, params.t_end, 2001)
    x_pred = model.predict_response(v0_test, tq)
    t_imp_pred = model.predict_impact_times(v0_test)

    print("x_pred shape:", x_pred.shape)
    print("first predicted impact times:", t_imp_pred[:10])
    model.save("PINN_free_free/parametric_model_artifacts")
    print("Saved to PINN_free_free/parametric_model_artifacts")
