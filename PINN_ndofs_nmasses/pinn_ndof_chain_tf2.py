"""
pinn_ndof_chain_tf2.py
----------------------
TensorFlow 2 port of pinn_ndof_chain.py  (same physics, same API).

Key differences from the TF1 version
--------------------------------------
- Eager execution by default — no placeholders, no Session, no tf.compat.v1
- Per-sample auto-diff via nested tf.GradientTape (replaces tf.gradients)
- Adam     : tf.keras.optimizers.Adam  (lr=1e-3, same default as TF1)
- L-BFGS-B : scipy.optimize.minimize wrapping TF2 gradient evaluation
             (equivalent to ScipyOptimizerInterface; same options/tolerances)
- GPU      : set CUDA_VISIBLE_DEVICES below; TF2 uses the GPU automatically
             when a CUDA device is visible.  Set to '-1' to force CPU.
- No graph-bloat: every PIPNNs instance is self-contained; training time
  per segment is constant regardless of how many segments have been run.

All public functions (find_impact_times, impact_velocity_update,
propagate_ics, newmark_beta) are identical to the TF1 version.

Author: Rui Zhang
"""

import os
import time
import numpy as np
import tensorflow as tf
from scipy.optimize import brentq, minimize

# ── GPU / CPU selection ────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # CPU: -1 | GPU0: 0
np.random.seed(1234)
tf.random.set_seed(1234)

# Adam: legacy path was removed in Keras 3 (TF ≥ 2.16); catch both
# AttributeError (path absent) and ImportError (path exists but raises).
try:
    _AdamClass = tf.keras.optimizers.legacy.Adam
except (AttributeError, ImportError):
    _AdamClass = tf.keras.optimizers.Adam


# ---------------------------------------------------------------------------
# Phase 1 — PINN (ODE + IC only)
# ---------------------------------------------------------------------------

class PIPNNs:
    """
    Parametric PINN for an n-DOF impact-damper chain  (TensorFlow 2 version).

    Network  : t  →  (x_1, ..., x_n)
    Loss     : beta_icx * L_ic  +  beta_fx * L_ode

    After training, call find_impact_times() to locate the first impact
    for every impactor via root-finding on the frozen network.

    Parameters  (identical to TF1 version)
    ----------
    lb, ub          : (1,) array-like — time-domain bounds for normalisation
    t0, t           : (1,1) and (N,1) — IC time and collocation times
    x0_total        : (1, n_dof)  — displacement ICs for primary DOFs
    xt0_total       : (1, n_dof)  — velocity ICs for primary DOFs
    y0              : (1, n_dof)  — IC positions of all n impactors
    yt0             : (1, n_dof)  — IC velocities of all n impactors
    M, K            : (n_dof, n_dof) — mass and stiffness matrices
    C               : (n_dof, n_dof) or None — damping matrix (0 if None)
    D               : float or (n_dof,) — impact gap for each DOF
    n_dof           : int — number of DOFs (= number of impactors)
    phi             : scalar — phase offset (accumulated segment start time)
    phi1, phi2      : scalars — forcing amplitude and angular frequency (rad/s)
                      Force on last DOF: phi1 * sin(phi2 * pi * (t + phi))
    layers          : list[int] — network architecture, e.g. [1, 64, 10]
    hyp_ini_weight_loss : (2,) — [beta_icx, beta_fx]
    optimizer_LB    : bool — run L-BFGS-B polishing step after Adam
    """

    def __init__(
        self,
        lb, ub,
        t0, t,
        x0_total, xt0_total,
        y0, yt0,
        M, K,
        D,
        n_dof,
        phi, phi1, phi2,
        layers,
        hyp_ini_weight_loss,
        C=None,
        optimizer_LB=True,
    ):
        # ── normalisation bounds (kept as tf.constant for _neural_net) ────
        self.lb = tf.constant(np.asarray(lb, dtype=np.float32).reshape(1, -1))
        self.ub = tf.constant(np.asarray(ub, dtype=np.float32).reshape(1, -1))

        # ── training data stored as tf.constant ───────────────────────────
        self.t_data   = tf.constant(np.asarray(t,  dtype=np.float32).reshape(-1, 1))
        self.t0_data  = tf.constant(np.asarray(t0, dtype=np.float32).reshape(-1, 1))
        self.x0_data  = tf.constant(np.asarray(x0_total,  dtype=np.float32))   # (1, n_dof)
        self.xt0_data = tf.constant(np.asarray(xt0_total, dtype=np.float32))   # (1, n_dof)

        # ── impactor ICs (numpy; used in find_impact_times / propagate) ───
        self.y0  = np.asarray(y0,  dtype=np.float64).flatten()
        self.yt0 = np.asarray(yt0, dtype=np.float64).flatten()

        self.n_dof = int(n_dof)

        D_arr = np.asarray(D, dtype=np.float64)
        self.D = float(D_arr) * np.ones(self.n_dof) if D_arr.ndim == 0 else D_arr.flatten()

        self.phi  = float(phi)
        self.phi1 = float(np.squeeze(phi1))
        self.phi2 = float(np.squeeze(phi2))

        self.beta_icx = float(hyp_ini_weight_loss[0])
        self.beta_fx  = float(hyp_ini_weight_loss[1])

        # ── system matrices ───────────────────────────────────────────────
        self.M_tf = tf.constant(M, dtype=tf.float32)
        self.K_tf = tf.constant(K, dtype=tf.float32)
        self.C_tf = tf.constant(
            np.zeros_like(M) if C is None else C, dtype=tf.float32
        )

        # ── network weights / biases ──────────────────────────────────────
        self.layers = layers
        self.weights, self.biases = self._init_NN(layers)
        self.trainable_vars = self.weights + self.biases

        # ── optimiser ─────────────────────────────────────────────────────
        self.adam = _AdamClass(learning_rate=1e-3)
        self._use_lbfgs = optimizer_LB

        # ── logs ──────────────────────────────────────────────────────────
        self.loss_log     = []
        self.loss_icx_log = []
        self.loss_fx_log  = []

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------
    def _init_NN(self, layers):
        weights, biases = [], []
        for l in range(len(layers) - 1):
            W = self._xavier([layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def _xavier(self, size):
        stddev = np.sqrt(2.0 / (size[0] + size[1]))
        return tf.Variable(
            tf.random.truncated_normal(size, stddev=stddev, dtype=tf.float32)
        )

    def _neural_net(self, t):
        H = 2.0 * (t - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]

    # ------------------------------------------------------------------
    # Physics (gradients via nested GradientTape)
    # ------------------------------------------------------------------
    def _net_u(self, t):
        """
        t  →  (x, x_t, x_tt)   each [N, n_dof]

        Uses nested persistent GradientTapes, mirroring the TF1 pattern:
            x_t[:, i]  = d(x[:, i]) / d(t)   for each DOF i
            x_tt[:, i] = d(x_t[:, i]) / d(t)
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(t)
                x = self._neural_net(t)          # [N, n_dof]
            # slice i:i+1 keeps the tensor 2-D ([N,1]) so gradient output
            # shape matches t ([N,1]) — identical to tf.gradients() in TF1
            x_t = tf.concat(
                [tape1.gradient(x[:, i:i+1], t) for i in range(self.n_dof)],
                axis=1,
            )   # [N, n_dof]
        x_tt = tf.concat(
            [tape2.gradient(x_t[:, i:i+1], t) for i in range(self.n_dof)],
            axis=1,
        )   # [N, n_dof]
        del tape1, tape2
        return x, x_t, x_tt

    def _net_f(self, t):
        """ODE residual  M x_tt + C x_t + K x - F(t)  →  [N, n_dof]"""
        x, x_t, x_tt = self._net_u(t)

        F = tf.concat(
            [tf.zeros_like(t)] * (self.n_dof - 1) +
            [self.phi1 * tf.sin(self.phi2 * np.pi * (t + self.phi))],
            axis=1,
        )   # [N, n_dof]

        residual = (
            tf.transpose(tf.matmul(self.M_tf, x_tt, transpose_b=True)) +
            tf.transpose(tf.matmul(self.C_tf, x_t,  transpose_b=True)) +
            tf.transpose(tf.matmul(self.K_tf, x,    transpose_b=True)) -
            F
        )
        return residual

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _compute_loss(self):
        """Returns (total_loss, loss_icx, loss_fx) as scalar tf.Tensors."""
        x0_pred, xt0_pred, _ = self._net_u(self.t0_data)
        loss_icx = (
            tf.reduce_mean(tf.square(x0_pred  - self.x0_data)) +
            tf.reduce_mean(tf.square(xt0_pred - self.xt0_data))
        )
        loss_fx = tf.reduce_mean(tf.square(self._net_f(self.t_data)))
        loss    = self.beta_icx * loss_icx + self.beta_fx * loss_fx
        return loss, loss_icx, loss_fx

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @tf.function
    def _train_step_adam(self):
        """Single Adam gradient step (compiled for speed)."""
        with tf.GradientTape() as tape:
            loss, loss_icx, loss_fx = self._compute_loss()
        grads = tape.gradient(loss, self.trainable_vars)
        self.adam.apply_gradients(zip(grads, self.trainable_vars))
        return loss, loss_icx, loss_fx

    # ---- L-BFGS-B helpers (scipy interface) --------------------------
    def _get_flat_params(self):
        return np.concatenate([v.numpy().flatten() for v in self.trainable_vars])

    def _set_flat_params(self, flat_f32):
        idx = 0
        for v in self.trainable_vars:
            n = int(np.prod(v.shape))
            v.assign(tf.reshape(flat_f32[idx:idx + n], v.shape))
            idx += n

    @tf.function
    def _loss_and_grads_tf(self):
        """Compiled loss + gradient computation used by the scipy wrapper."""
        with tf.GradientTape() as tape:
            loss, loss_icx, loss_fx = self._compute_loss()
        grads = tape.gradient(loss, self.trainable_vars)
        return loss, loss_icx, loss_fx, grads

    def _lbfgs_objective(self, flat_params_f64):
        """
        Callable for scipy.optimize.minimize(method='L-BFGS-B', jac=True).
        Accepts and returns float64 arrays as required by scipy.
        """
        self._set_flat_params(flat_params_f64.astype(np.float32))
        loss, loss_icx, loss_fx, grads = self._loss_and_grads_tf()

        lv  = float(loss)
        lic = float(loss_icx)
        lfx = float(loss_fx)
        self.loss_log.append(lv)
        self.loss_icx_log.append(lic)
        self.loss_fx_log.append(lfx)
        print('Loss: %.5e   Loss_ic: %.5e   Loss_ode: %.5e' % (lv, lic, lfx))

        grad_flat = np.concatenate(
            [g.numpy().flatten() for g in grads]
        ).astype(np.float64)
        return lv, grad_flat

    def train(self, nIter=1000, optimizer_LB=True, print_every=100):
        """
        Phase 1: train NN on ODE + IC loss.
        After this, call find_impact_times() for Phase 2.
        """
        t0_wall = time.time()
        for it in range(nIter):
            loss, loss_icx, loss_fx = self._train_step_adam()
            if it % print_every == 0:
                lv  = float(loss)
                lic = float(loss_icx)
                lfx = float(loss_fx)
                self.loss_log.append(lv)
                self.loss_icx_log.append(lic)
                self.loss_fx_log.append(lfx)
                print('It %5d  Loss %.3e  Loss_ic %.3e  Loss_ode %.3e  %.2fs'
                      % (it, lv, lic, lfx, time.time() - t0_wall))
                t0_wall = time.time()

        if optimizer_LB and self._use_lbfgs:
            x0_flat = self._get_flat_params().astype(np.float64)
            minimize(
                self._lbfgs_objective,
                x0_flat,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 50000,
                    'maxfun':  50000,
                    'maxcor':  50,
                    'maxls':   50,
                    'ftol':    np.finfo(float).eps,
                    'gtol':    1e-8,
                },
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, t):
        """
        Forward pass at query times t (shape [N, 1] or scalar).

        Returns
        -------
        x, xt, xtt : each (N, n_dof)  as numpy arrays
        """
        t_tf = tf.constant(np.asarray(t, dtype=np.float32).reshape(-1, 1))
        x, x_t, x_tt = self._net_u(t_tf)
        return x.numpy(), x_t.numpy(), x_tt.numpy()


# ---------------------------------------------------------------------------
# Phase 2 — root-finding for impact times (all n DOFs)
# (identical to TF1 version — only calls model.predict())
# ---------------------------------------------------------------------------

def find_impact_times(
    model,
    y0, yt0,
    D,
    T_max,
    n_scan=500,
    tol=1e-8,
):
    """
    Phase 2: find the first impact time for every DOF on the frozen network.

    Strategy
    --------
    For each DOF i:
      1. Evaluate the gap function over n_scan evenly-spaced points in (0, T_max]:
             gap_i(t) = |x_i(t) - y_i(t)| - D_i
         where  y_i(t) = y0_i + yt0_i * t  (free flight).
      2. Detect the first upward zero-crossing (gap: negative → positive).
      3. Refine with Brent's method between the two bracketing scan points.

    Returns
    -------
    t_impacts : (n_dof,) float array — impact time for each DOF
    hit       : (n_dof,) bool array  — True when a genuine impact was found
    """
    n = model.n_dof
    y0  = np.asarray(y0,  dtype=np.float64).flatten()
    yt0 = np.asarray(yt0, dtype=np.float64).flatten()
    D_arr = np.asarray(D, dtype=np.float64)
    D_vec = float(D_arr) * np.ones(n) if D_arr.ndim == 0 else D_arr.flatten()
    T_max = float(T_max)

    t_scan = np.linspace(1e-4 * T_max, T_max, n_scan).reshape(-1, 1).astype(np.float32)
    x_scan, _, _ = model.predict(t_scan)
    t_flat = t_scan.flatten().astype(np.float64)

    t_impacts = np.full(n, T_max)
    hit       = np.zeros(n, dtype=bool)

    for i in range(n):
        y_scan = y0[i] + yt0[i] * t_flat
        gap    = np.abs(x_scan[:, i] - y_scan) - D_vec[i]

        neg_idx = np.where(gap < 0)[0]
        if len(neg_idx) == 0:
            print(f"  DOF {i+1}: gap never closed — no impact in [0, {T_max:.3f}]")
            continue

        gap_tail = gap[neg_idx[0]:]
        up_cross = np.where(np.diff(np.sign(gap_tail)) > 0)[0]
        if len(up_cross) == 0:
            print(f"  DOF {i+1}: masses separated but never re-collided")
            continue

        bracket_i = neg_idx[0] + up_cross[0]
        ta = float(t_flat[bracket_i])
        tb = float(t_flat[bracket_i + 1])

        def _gap(t_val, _i=i, _y0=y0[i], _yt0=yt0[i], _D=D_vec[i]):
            xv, _, _ = model.predict(np.array([[t_val]], dtype=np.float32))
            yv = _y0 + _yt0 * t_val
            return abs(float(xv[0, _i]) - yv) - _D

        try:
            t_imp = brentq(_gap, ta, tb, xtol=tol)
        except ValueError:
            t_imp = ta

        print(f"  DOF {i+1}: impact at t = {t_imp:.6f}")
        t_impacts[i] = t_imp
        hit[i]       = True

    return t_impacts, hit


# ---------------------------------------------------------------------------
# Velocity update at impact
# ---------------------------------------------------------------------------

def impact_velocity_update(mx, my, r, xt_minus, yt_minus):
    """
    Post-impact velocities via 1-D momentum conservation + restitution.
    Returns (xt_plus, yt_plus).
    """
    mx, my = float(mx), float(my)
    xt_m, yt_m = float(xt_minus), float(yt_minus)
    total = mx + my
    xt_p  = ((mx - r * my) * xt_m + my * (1.0 + r) * yt_m) / total
    yt_p  = (mx * (1.0 + r) * xt_m + (my - r * mx) * yt_m) / total
    return xt_p, yt_p


# ---------------------------------------------------------------------------
# IC propagation helper
# ---------------------------------------------------------------------------

def propagate_ics(
    model,
    t_impact,
    x0_post, xt0_post,
    y0, yt0,
    hit_dof,
    mx_list, my_list, r,
    A_inv_B,
):
    """
    Compute initial conditions for the next segment after an impact event.
    (Identical to TF1 version.)
    """
    t = float(t_impact)

    x0_next  = x0_post.copy()
    xt0_next = xt0_post.copy()

    y0_next  = y0  + yt0  * t
    yt0_next = yt0.copy()

    j = hit_dof
    xt_m = float(xt0_post[0, j])
    yt_m = float(yt0[j])

    V0 = np.array([[xt_m], [yt_m]])
    V1 = A_inv_B @ V0
    xt0_next[0, j] = float(V1[0])
    yt0_next[j]    = float(V1[1])

    return x0_next, xt0_next, y0_next, yt0_next


# ---------------------------------------------------------------------------
# Newmark-beta integrator (reference solution)
# ---------------------------------------------------------------------------

def newmark_beta(
    M, C, K, F,
    dt, n_steps, n_dof,
    x0=None, xt0=None,
    beta=0.25, gamma=0.5,
):
    """
    Newmark-beta implicit integrator.  (Identical to TF1 version.)
    """
    x   = np.zeros((n_dof, n_steps))
    xt  = np.zeros((n_dof, n_steps))
    xtt = np.zeros((n_dof, n_steps))

    if x0  is not None: x[:,  0] = np.asarray(x0).flatten()
    if xt0 is not None: xt[:, 0] = np.asarray(xt0).flatten()

    xtt[:, 0] = np.linalg.solve(M, F[:, 0] - C @ xt[:, 0] - K @ x[:, 0])

    K_eff     = M / (beta * dt**2) + gamma * C / (beta * dt) + K
    K_eff_inv = np.linalg.inv(K_eff)

    for i in range(1, n_steps):
        F_eff = (
            F[:, i]
            + M @ (x[:, i-1] / (beta * dt**2) + xt[:, i-1] / (beta * dt)
                   + xtt[:, i-1] * (0.5/beta - 1))
            + C @ (gamma * x[:, i-1] / (beta * dt)
                   - xt[:, i-1] * (1 - gamma/beta)
                   - dt * xtt[:, i-1] * (1 - gamma / (2*beta)))
        )
        x[:, i]   = K_eff_inv @ F_eff
        xtt[:, i] = (x[:, i] - x[:, i-1]) / (beta * dt**2) \
                    - xt[:, i-1] / (beta * dt) \
                    - xtt[:, i-1] * (0.5/beta - 1)
        xt[:, i]  = xt[:, i-1] + dt * ((1 - gamma) * xtt[:, i-1] + gamma * xtt[:, i])

    return x, xt, xtt
