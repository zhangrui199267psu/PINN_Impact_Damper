"""
Parametric PINN for impact-driven response over many impacts with ONE network.

Goal
----
Learn a single mapping
    (t, phi1, phi2) -> x(t; phi1, phi2)
for phi1 in [1, 2] and phi2 in [10, 20], over a long horizon containing
~50 impacts. The model is trained on sampled parameter pairs and can be used
for unseen in-range pairs.

Key idea
--------
To avoid segment-by-segment retraining, we use a smooth contact-force surrogate:

    m x_tt + c x_t + k x + f_contact(x, x_t) = phi1 * sin(phi2 * pi * t)

where f_contact is a differentiable penalty active near the impact gap. This
allows one global PINN to represent all impacts over the full time horizon.

Notes
-----
- This script is intentionally lightweight and self-contained.
- If your original model uses explicit impactor states, you can extend the
  network output to include impactor position/velocity and swap in your exact
  contact law.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf


@dataclass
class TrainingConfig:
    phi1_range: Tuple[float, float] = (1.0, 2.0)
    phi2_range: Tuple[float, float] = (10.0, 20.0)
    n_param_samples: int = 20
    n_collocation_t: int = 2500
    t_final: float = 30.0
    # System coefficients
    m: float = 1.0
    c: float = 0.02
    k: float = 80.0
    gap: float = 0.08
    restitution_proxy: float = 0.08
    contact_stiffness: float = 7000.0
    contact_smoothness: float = 80.0
    # Optimizer
    lr: float = 1e-3
    adam_epochs: int = 8000
    log_every: int = 500
    # Network
    layers: Tuple[int, ...] = (3, 128, 128, 128, 1)


class ParametricImpactPINN:
    """One-network parametric PINN on (t, phi1, phi2) for long-horizon impacts."""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.model = self._build_mlp(cfg.layers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

        # Normalize inputs to [-1, 1]
        self.lb = tf.constant([0.0, cfg.phi1_range[0], cfg.phi2_range[0]], dtype=tf.float32)
        self.ub = tf.constant([cfg.t_final, cfg.phi1_range[1], cfg.phi2_range[1]], dtype=tf.float32)

    @staticmethod
    def _build_mlp(layers: Tuple[int, ...]) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(layers[0],), dtype=tf.float32)
        z = inputs
        for w in layers[1:-1]:
            z = tf.keras.layers.Dense(w, activation="tanh")(z)
        outputs = tf.keras.layers.Dense(layers[-1], activation=None)(z)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _normalize(self, x: tf.Tensor) -> tf.Tensor:
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

    def net_x(self, t: tf.Tensor, phi1: tf.Tensor, phi2: tf.Tensor) -> tf.Tensor:
        inp = tf.concat([t, phi1, phi2], axis=1)
        return self.model(self._normalize(inp))

    def contact_force(self, x: tf.Tensor, xt: tf.Tensor) -> tf.Tensor:
        """
        Symmetric soft contact walls at x = ±gap.
        Smooth approximation produces impact-like impulsive responses while
        remaining differentiable for PINN training.
        """
        cfg = self.cfg
        over_pos = tf.nn.softplus(cfg.contact_smoothness * (x - cfg.gap)) / cfg.contact_smoothness
        over_neg = tf.nn.softplus(cfg.contact_smoothness * (-x - cfg.gap)) / cfg.contact_smoothness

        # Dashpot proxy opposing motion near contact; keeps rebounds stable.
        damping_term = cfg.restitution_proxy * xt

        return cfg.contact_stiffness * (over_pos - over_neg) + cfg.contact_stiffness * 0.02 * damping_term * (
            tf.cast(over_pos > 0.0, tf.float32) + tf.cast(over_neg > 0.0, tf.float32)
        )

    @tf.function
    def physics_residual(self, t: tf.Tensor, phi1: tf.Tensor, phi2: tf.Tensor):
        with tf.GradientTape() as tape2:
            tape2.watch(t)
            with tf.GradientTape() as tape1:
                tape1.watch(t)
                x = self.net_x(t, phi1, phi2)
            xt = tape1.gradient(x, t)
        xtt = tape2.gradient(xt, t)

        forcing = phi1 * tf.sin(phi2 * np.pi * t)
        f_contact = self.contact_force(x, xt)

        cfg = self.cfg
        residual = cfg.m * xtt + cfg.c * xt + cfg.k * x + f_contact - forcing

        # Initial condition priors at t = 0 for every sampled parameter pair
        x0 = self.net_x(tf.zeros_like(t), phi1, phi2)
        ic_loss = tf.reduce_mean(tf.square(x0))

        return residual, ic_loss, x, xt

    def sample_training_points(self):
        cfg = self.cfg

        # Latin-hypercube-like random sampling in parameter domain
        phi1 = np.random.uniform(cfg.phi1_range[0], cfg.phi1_range[1], (cfg.n_param_samples, 1)).astype(np.float32)
        phi2 = np.random.uniform(cfg.phi2_range[0], cfg.phi2_range[1], (cfg.n_param_samples, 1)).astype(np.float32)

        t = np.random.uniform(0.0, cfg.t_final, (cfg.n_collocation_t, 1)).astype(np.float32)

        # Tile all sampled parameter points against collocation times
        t_all = np.repeat(t, cfg.n_param_samples, axis=0)
        phi1_all = np.tile(phi1, (cfg.n_collocation_t, 1))
        phi2_all = np.tile(phi2, (cfg.n_collocation_t, 1))

        return (
            tf.convert_to_tensor(t_all),
            tf.convert_to_tensor(phi1_all),
            tf.convert_to_tensor(phi2_all),
        )

    @tf.function
    def train_step(self, t, phi1, phi2):
        with tf.GradientTape() as tape:
            residual, ic_loss, _, _ = self.physics_residual(t, phi1, phi2)
            pde_loss = tf.reduce_mean(tf.square(residual))
            loss = pde_loss + 50.0 * ic_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, pde_loss, ic_loss

    def train(self):
        t, phi1, phi2 = self.sample_training_points()
        for epoch in range(1, self.cfg.adam_epochs + 1):
            loss, pde, ic = self.train_step(t, phi1, phi2)
            if epoch % self.cfg.log_every == 0 or epoch == 1:
                print(
                    f"epoch={epoch:5d} | total={float(loss):.3e} | "
                    f"pde={float(pde):.3e} | ic={float(ic):.3e}"
                )

    def predict_response(self, t_query: np.ndarray, phi1: float, phi2: float) -> np.ndarray:
        t_query = np.asarray(t_query, dtype=np.float32).reshape(-1, 1)
        p1 = np.full_like(t_query, phi1)
        p2 = np.full_like(t_query, phi2)
        x = self.net_x(tf.constant(t_query), tf.constant(p1), tf.constant(p2))
        return x.numpy().reshape(-1)

    def predict_contact_indicator(self, t_query: np.ndarray, phi1: float, phi2: float) -> np.ndarray:
        t_query = np.asarray(t_query, dtype=np.float32).reshape(-1, 1)
        p1 = np.full_like(t_query, phi1)
        p2 = np.full_like(t_query, phi2)

        t_tf = tf.constant(t_query)
        with tf.GradientTape() as tape:
            tape.watch(t_tf)
            x = self.net_x(t_tf, tf.constant(p1), tf.constant(p2))
        xt = tape.gradient(x, t_tf)

        # Indicator spikes near impacts
        fc = self.contact_force(x, xt)
        return np.abs(fc.numpy().reshape(-1))

    def extract_impact_times(
        self,
        t_query: np.ndarray,
        phi1: float,
        phi2: float,
        n_impacts: int = 50,
        threshold_quantile: float = 0.985,
    ) -> np.ndarray:
        """Return estimated impact times from peaks of contact indicator."""
        indicator = self.predict_contact_indicator(t_query, phi1, phi2)
        t_query = np.asarray(t_query).reshape(-1)

        threshold = np.quantile(indicator, threshold_quantile)
        peak_mask = np.zeros_like(indicator, dtype=bool)
        peak_mask[1:-1] = (indicator[1:-1] > indicator[:-2]) & (indicator[1:-1] > indicator[2:])
        peak_mask &= indicator >= threshold

        impact_times = t_query[peak_mask]
        if impact_times.size > n_impacts:
            impact_times = impact_times[:n_impacts]
        return impact_times


if __name__ == "__main__":
    np.random.seed(1234)
    tf.random.set_seed(1234)

    cfg = TrainingConfig(
        phi1_range=(1.0, 2.0),
        phi2_range=(10.0, 20.0),
        n_param_samples=20,
        n_collocation_t=2000,
        t_final=30.0,
        adam_epochs=4000,
        log_every=200,
    )

    pinn = ParametricImpactPINN(cfg)
    pinn.train()

    # Unseen in-range parameter query
    phi1_test, phi2_test = 1.37, 16.25
    t_eval = np.linspace(0.0, cfg.t_final, 12000)

    x_pred = pinn.predict_response(t_eval, phi1_test, phi2_test)
    impact_times = pinn.extract_impact_times(t_eval, phi1_test, phi2_test, n_impacts=50)

    print(f"Predicted response points: {x_pred.shape[0]}")
    print(f"Detected impacts: {impact_times.shape[0]}")
    print("First 10 impact times:", impact_times[:10])
