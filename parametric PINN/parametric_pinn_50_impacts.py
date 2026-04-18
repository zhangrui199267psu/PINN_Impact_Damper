"""
Parametric PINN for full 50-impact horizons while preserving the original
`pinn_ndof_chain_tf2.py` workflow and training settings.

Design (keeps original ideas/settings)
-------------------------------------
1) For each sampled (phi1, phi2), generate a full 50-impact trajectory with the
   original segment-by-segment solver `PIPNNs` (Adam + optional L-BFGS), using
   the same physical/optimization defaults from `pinn_ndof_chain_sim_tf2.ipynb`.
2) Train ONE parametric network on aggregated trajectories:
      (t, phi1, phi2) -> [x(t), y(t)]
   where x: primary DOFs, y: impactor DOFs.
3) For unseen parameters, infer x(t;phi1,phi2) over the whole horizon and
   extract impact times from predicted gap crossings |x-y|-D.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy.optimize import brentq, minimize

from PINN.pinn_ndof_chain_tf2 import PIPNNs, find_impact_times, propagate_ics


# -----------------------------------------------------------------------------
# Configs (matching the original notebook defaults)
# -----------------------------------------------------------------------------

@dataclass
class LegacySimConfig:
    n_dof: int = 20
    m_x: float = 1.0
    m_y: float = 0.3
    k: float = 1.0
    c: float = 0.0
    D: float = 1.0
    r: float = 1.0

    # Original PINN per-segment settings
    num_neurons: int = 64
    hyp_ini_weight_loss: Tuple[float, float] = (1.0, 1.0)
    optimizer_LB_value: bool = True

    # Segment workflow (50 impacts)
    n_segments: int = 50
    T_seg: float = 1.0
    num_seg: int = 1000
    nIter_seg: int = 1000


@dataclass
class ParametricDataConfig:
    phi1_range: Tuple[float, float] = (1.0, 2.0)
    phi2_range: Tuple[float, float] = (10.0, 20.0)
    n_param_samples: int = 20
    seed: int = 1234


@dataclass
class ParametricModelConfig:
    layers: Tuple[int, ...] = (3, 128, 128, 128, 40)  # 40 = 2*n_dof for n_dof=20
    lr: float = 1e-3
    adam_epochs: int = 5000
    log_every: int = 200
    optimizer_LB: bool = True
    # Loss weights
    beta_data: float = 1.0
    beta_phys: float = 1.0
    beta_ic: float = 1.0


# -----------------------------------------------------------------------------
# Legacy system helpers
# -----------------------------------------------------------------------------


def build_chain_matrices(cfg: LegacySimConfig):
    n = cfg.n_dof
    M = np.diag(cfg.m_x * np.ones(n))

    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0 * cfg.k if 0 < i < n - 1 else cfg.k
        if i > 0:
            K[i, i - 1] = -cfg.k
        if i < n - 1:
            K[i, i + 1] = -cfg.k

    C = (cfg.c / cfg.k) * K if cfg.k != 0 else np.zeros_like(K)
    return M, C, K


def _impact_update_matrix(mx: float, my: float, r: float):
    A = np.array([[mx, my], [-1.0, 1.0]], dtype=float)
    B = np.array([[mx, my], [r, -r]], dtype=float)
    return np.linalg.inv(A) @ B


class LegacyFullHorizonGenerator:
    """Generate full 50-impact responses with original segment-by-segment PINN."""

    def __init__(self, sim_cfg: LegacySimConfig):
        self.cfg = sim_cfg
        self.M, self.C, self.K = build_chain_matrices(sim_cfg)
        self.layers = [1, sim_cfg.num_neurons, sim_cfg.n_dof]
        self.lb = np.array([0.0])
        self.ub = np.array([sim_cfg.T_seg])
        self.A_inv_B = _impact_update_matrix(sim_cfg.m_x, sim_cfg.m_y, sim_cfg.r)

    def run_case(self, phi1: float, phi2: float) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        n = cfg.n_dof

        t0 = np.array([[0.0]])
        cur_x0 = np.zeros((1, n))
        cur_xt0 = np.zeros((1, n))
        cur_y0 = np.zeros(n)
        cur_yt0 = np.zeros(n)
        cur_phi = np.array([[0.0]])

        t_cumulative = 0.0
        all_t, all_x, all_y = [], [], []
        all_event_time = []

        for seg in range(1, cfg.n_segments + 1):
            t_seg_arr = np.linspace(0.0, cfg.T_seg, cfg.num_seg).reshape(-1, 1)

            model = PIPNNs(
                self.lb,
                self.ub,
                t0,
                t_seg_arr,
                cur_x0,
                cur_xt0,
                cur_y0,
                cur_yt0,
                self.M,
                self.K,
                cfg.D,
                n,
                cur_phi,
                float(phi1),
                float(phi2),
                self.layers,
                np.array(cfg.hyp_ini_weight_loss),
                C=self.C,
                optimizer_LB=cfg.optimizer_LB_value,
            )
            model.train(nIter=cfg.nIter_seg, optimizer_LB=cfg.optimizer_LB_value)

            t_impacts, hit = find_impact_times(
                model,
                cur_y0,
                cur_yt0,
                cfg.D,
                T_max=cfg.T_seg,
            )
            t_impact = float(np.min(t_impacts))
            hit_dof = int(np.argmin(t_impacts))
            if not np.any(hit):
                # If no impact found, still roll out full segment and stop
                t_impact = float(cfg.T_seg)
                hit_dof = 0

            t_sim = np.linspace(0.0, t_impact, cfg.num_seg + 1).reshape(-1, 1)
            x_sim, xt_sim, _ = model.predict(t_sim)
            y_sim = (cur_y0 + cur_yt0 * t_sim).T  # (n_dof, num_seg+1)

            all_t.append(t_sim.flatten() + t_cumulative)
            all_x.append(x_sim)
            all_y.append(y_sim.T)
            all_event_time.append(t_cumulative + t_impact)

            x_at_imp, xt_at_imp, _ = model.predict(np.array([[t_impact]]))
            cur_x0, cur_xt0, cur_y0, cur_yt0 = propagate_ics(
                model,
                t_impact,
                x_at_imp,
                xt_at_imp,
                cur_y0,
                cur_yt0,
                hit_dof,
                cfg.m_x * np.ones(n),
                cfg.m_y * np.ones(n),
                cfg.r,
                self.A_inv_B,
            )
            cur_phi = cur_phi + t_impact
            t_cumulative += t_impact

        return {
            "t": np.concatenate(all_t),
            "x": np.vstack(all_x),
            "y": np.vstack(all_y),
            "impact_events": np.asarray(all_event_time),
            "phi1": float(phi1),
            "phi2": float(phi2),
        }


# -----------------------------------------------------------------------------
# Parametric one-network model (Adam + optional L-BFGS)
# -----------------------------------------------------------------------------


class ParametricFullHorizonPINN:
    def __init__(self, sim_cfg: LegacySimConfig, model_cfg: ParametricModelConfig):
        self.sim_cfg = sim_cfg
        self.model_cfg = model_cfg

        out_dim = 2 * sim_cfg.n_dof
        if model_cfg.layers[-1] != out_dim:
            layers = list(model_cfg.layers)
            layers[-1] = out_dim
            self.layers = tuple(layers)
        else:
            self.layers = model_cfg.layers

        self.net = self._build_mlp(self.layers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=model_cfg.lr)

        self.M, self.C, self.K = build_chain_matrices(sim_cfg)
        self.M_tf = tf.constant(self.M, dtype=tf.float32)
        self.C_tf = tf.constant(self.C, dtype=tf.float32)
        self.K_tf = tf.constant(self.K, dtype=tf.float32)

        self.lb = None
        self.ub = None

    @staticmethod
    def _build_mlp(layers: Tuple[int, ...]):
        inp = tf.keras.Input(shape=(layers[0],), dtype=tf.float32)
        z = inp
        for width in layers[1:-1]:
            z = tf.keras.layers.Dense(width, activation="tanh")(z)
        out = tf.keras.layers.Dense(layers[-1], activation=None)(z)
        return tf.keras.Model(inp, out)

    def _set_normalization_bounds(self, t_max: float, phi1_rng, phi2_rng):
        self.lb = tf.constant([0.0, phi1_rng[0], phi2_rng[0]], dtype=tf.float32)
        self.ub = tf.constant([t_max, phi1_rng[1], phi2_rng[1]], dtype=tf.float32)

    def _norm(self, z: tf.Tensor) -> tf.Tensor:
        return 2.0 * (z - self.lb) / (self.ub - self.lb) - 1.0

    def _forward(self, t: tf.Tensor, phi1: tf.Tensor, phi2: tf.Tensor):
        inp = tf.concat([t, phi1, phi2], axis=1)
        pred = self.net(self._norm(inp))
        n = self.sim_cfg.n_dof
        x = pred[:, :n]
        y = pred[:, n:]
        return x, y

    @tf.function
    def _loss_components(self, t, phi1, phi2, x_data, y_data, t0_mask):
        n = self.sim_cfg.n_dof

        with tf.GradientTape() as tape2:
            tape2.watch(t)
            with tf.GradientTape() as tape1:
                tape1.watch(t)
                x_pred, y_pred = self._forward(t, phi1, phi2)
            x_t = tf.squeeze(tape1.batch_jacobian(x_pred, t), axis=-1)  # [N, n]
        x_tt = tf.squeeze(tape2.batch_jacobian(x_t, t), axis=-1)  # [N, n]

        forcing = tf.concat(
            [tf.zeros_like(t)] * (n - 1) + [phi1 * tf.sin(phi2 * np.pi * t)],
            axis=1,
        )

        residual = (
            tf.transpose(tf.matmul(self.M_tf, x_tt, transpose_b=True))
            + tf.transpose(tf.matmul(self.C_tf, x_t, transpose_b=True))
            + tf.transpose(tf.matmul(self.K_tf, x_pred, transpose_b=True))
            - forcing
        )

        data_loss = tf.reduce_mean(tf.square(x_pred - x_data)) + tf.reduce_mean(tf.square(y_pred - y_data))
        phys_loss = tf.reduce_mean(tf.square(residual))

        # IC consistency on points with local-segment t = 0
        x0_pred = tf.boolean_mask(x_pred, t0_mask[:, 0])
        y0_pred = tf.boolean_mask(y_pred, t0_mask[:, 0])
        ic_loss = tf.reduce_mean(tf.square(x0_pred)) + tf.reduce_mean(tf.square(y0_pred))
        return data_loss, phys_loss, ic_loss

    @tf.function
    def _adam_step(self, t, phi1, phi2, x_data, y_data, t0_mask):
        with tf.GradientTape() as tape:
            d, p, i = self._loss_components(t, phi1, phi2, x_data, y_data, t0_mask)
            loss = self.model_cfg.beta_data * d + self.model_cfg.beta_phys * p + self.model_cfg.beta_ic * i
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
        return loss, d, p, i

    def _get_flat_params(self):
        return np.concatenate([v.numpy().ravel() for v in self.net.trainable_variables])

    def _set_flat_params(self, flat):
        idx = 0
        for v in self.net.trainable_variables:
            n = int(np.prod(v.shape))
            v.assign(np.reshape(flat[idx:idx + n], v.shape))
            idx += n

    def _lbfgs_objective(self, flat, t, phi1, phi2, x_data, y_data, t0_mask):
        self._set_flat_params(flat.astype(np.float32))
        with tf.GradientTape() as tape:
            d, p, i = self._loss_components(t, phi1, phi2, x_data, y_data, t0_mask)
            loss = self.model_cfg.beta_data * d + self.model_cfg.beta_phys * p + self.model_cfg.beta_ic * i
        grads = tape.gradient(loss, self.net.trainable_variables)
        gflat = np.concatenate([g.numpy().ravel() for g in grads]).astype(np.float64)
        return float(loss.numpy()), gflat

    def train(self, dataset: Dict[str, np.ndarray], phi1_rng, phi2_rng):
        t = dataset["t_local"].astype(np.float32).reshape(-1, 1)
        phi1 = dataset["phi1"].astype(np.float32).reshape(-1, 1)
        phi2 = dataset["phi2"].astype(np.float32).reshape(-1, 1)
        x_data = dataset["x"].astype(np.float32)
        y_data = dataset["y"].astype(np.float32)
        t0_mask = dataset["t_local"].reshape(-1, 1) < 1e-12

        self._set_normalization_bounds(float(np.max(t)), phi1_rng, phi2_rng)

        t_tf = tf.constant(t)
        phi1_tf = tf.constant(phi1)
        phi2_tf = tf.constant(phi2)
        x_tf = tf.constant(x_data)
        y_tf = tf.constant(y_data)
        t0_mask_tf = tf.constant(t0_mask)

        for ep in range(1, self.model_cfg.adam_epochs + 1):
            total, d, p, i = self._adam_step(t_tf, phi1_tf, phi2_tf, x_tf, y_tf, t0_mask_tf)
            if ep == 1 or ep % self.model_cfg.log_every == 0:
                print(
                    f"ep={ep:5d} total={float(total):.3e} "
                    f"data={float(d):.3e} phys={float(p):.3e} ic={float(i):.3e}"
                )

        if self.model_cfg.optimizer_LB:
            x0 = self._get_flat_params().astype(np.float64)
            minimize(
                lambda z: self._lbfgs_objective(z, t_tf, phi1_tf, phi2_tf, x_tf, y_tf, t0_mask_tf),
                x0,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": 50000,
                    "maxfun": 50000,
                    "maxcor": 50,
                    "maxls": 50,
                    "ftol": np.finfo(float).eps,
                    "gtol": 1e-8,
                },
            )

    def predict_xy(self, t_query: np.ndarray, phi1: float, phi2: float):
        t = np.asarray(t_query, dtype=np.float32).reshape(-1, 1)
        p1 = np.full_like(t, float(phi1))
        p2 = np.full_like(t, float(phi2))
        x, y = self._forward(tf.constant(t), tf.constant(p1), tf.constant(p2))
        return x.numpy(), y.numpy()

    def predict_x(self, t_query: np.ndarray, phi1: float, phi2: float):
        x, _ = self.predict_xy(t_query, phi1, phi2)
        return x

    def extract_impact_times(self, t_query: np.ndarray, phi1: float, phi2: float, D: float, max_impacts: int = 50):
        x, y = self.predict_xy(t_query, phi1, phi2)
        t = np.asarray(t_query).reshape(-1)
        gap = np.max(np.abs(x - y) - D, axis=1)

        idx = np.where((gap[:-1] < 0.0) & (gap[1:] >= 0.0))[0]
        t_imp = []
        for i in idx:
            ta, tb = t[i], t[i + 1]
            ga, gb = gap[i], gap[i + 1]
            if np.isclose(gb, ga):
                t_imp.append(ta)
            else:
                t_imp.append(ta - ga * (tb - ta) / (gb - ga))
            if len(t_imp) >= max_impacts:
                break
        return np.asarray(t_imp)


# -----------------------------------------------------------------------------
# Dataset assembly
# -----------------------------------------------------------------------------


def sample_parameter_pairs(cfg: ParametricDataConfig):
    rng = np.random.default_rng(cfg.seed)
    p1 = rng.uniform(cfg.phi1_range[0], cfg.phi1_range[1], cfg.n_param_samples)
    p2 = rng.uniform(cfg.phi2_range[0], cfg.phi2_range[1], cfg.n_param_samples)
    return np.stack([p1, p2], axis=1)


def build_parametric_dataset(generator: LegacyFullHorizonGenerator, data_cfg: ParametricDataConfig):
    pairs = sample_parameter_pairs(data_cfg)

    all_t_local = []
    all_phi1 = []
    all_phi2 = []
    all_x = []
    all_y = []
    cases = []

    for i, (phi1, phi2) in enumerate(pairs, start=1):
        print(f"\n[case {i}/{len(pairs)}] generating legacy trajectory for phi1={phi1:.4f}, phi2={phi2:.4f}")
        out = generator.run_case(float(phi1), float(phi2))
        t_abs = out["t"]
        # convert to local-in-horizon [0, T_total] query time for parametric net
        t_local = t_abs - t_abs.min()

        all_t_local.append(t_local)
        all_phi1.append(np.full_like(t_local, phi1, dtype=float))
        all_phi2.append(np.full_like(t_local, phi2, dtype=float))
        all_x.append(out["x"])
        all_y.append(out["y"])
        cases.append(out)

    dataset = {
        "t_local": np.concatenate(all_t_local),
        "phi1": np.concatenate(all_phi1),
        "phi2": np.concatenate(all_phi2),
        "x": np.vstack(all_x),
        "y": np.vstack(all_y),
        "cases": cases,
    }
    return dataset


if __name__ == "__main__":
    np.random.seed(1234)
    tf.random.set_seed(1234)

    sim_cfg = LegacySimConfig()
    data_cfg = ParametricDataConfig(
        phi1_range=(1.0, 2.0),
        phi2_range=(10.0, 20.0),
        n_param_samples=20,
    )
    model_cfg = ParametricModelConfig(
        layers=(3, 128, 128, 128, 2 * sim_cfg.n_dof),
        adam_epochs=2000,
        optimizer_LB=True,
    )

    generator = LegacyFullHorizonGenerator(sim_cfg)
    dataset = build_parametric_dataset(generator, data_cfg)

    model = ParametricFullHorizonPINN(sim_cfg, model_cfg)
    model.train(dataset, data_cfg.phi1_range, data_cfg.phi2_range)

    # Unseen in-range query
    phi1_test, phi2_test = 1.37, 16.25
    t_eval = np.linspace(0.0, np.max(dataset["t_local"]), 20000)

    x_pred = model.predict_x(t_eval, phi1_test, phi2_test)
    t_imp = model.extract_impact_times(t_eval, phi1_test, phi2_test, D=sim_cfg.D, max_impacts=50)

    print("x_pred shape:", x_pred.shape)
    print("detected impacts:", len(t_imp))
    print("first 10 impacts:", t_imp[:10])
