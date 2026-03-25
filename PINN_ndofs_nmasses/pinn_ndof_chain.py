"""
pinn_ndof_chain.py
------------------
Two-phase PINN for an n-DOF chain where EVERY DOF has its own internal
impact damper (one impactor per DOF).

System
------
n DOFs:  x1, x2, ..., xn  (linear spring-coupled chain)
  - DOF i (index i-1): primary mass m_i, internal impactor y_i (free-flight)
  - DOF n (last): also carries the external harmonic excitation

EOM between impacts (no contact):
    M x_tt + C x_t + K x = F(t)
    y_i(t) = y0_i + yt0_i * t       (free flight for each impactor i)

Impact condition for DOF i:
    |x_i(t*) - y_i(t*)| = D_i

Two-phase strategy
------------------
Phase 1  — PIPNNs.train()
    Minimise:  beta_icx * L_ic + beta_fx * L_ode
    Network learns the physical trajectory for the full segment window.
    Network weights are frozen after this phase.

Phase 2  — find_impact_times()
    For each DOF i independently, root-find t*_i on the frozen network:
        gap_i(t) = |x_i(t) - y_i(t)| - D_i  = 0
    The segment ends at t_seg = min_i(t*_i).
    The DOF j with t*_j = t_seg undergoes an impact-velocity update.

Author: Rui Zhang
"""

import os
import time
import numpy as np
import tensorflow as tf
from scipy.optimize import brentq

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # CPU: -1 | GPU0: 0
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

tf.compat.v1.disable_eager_execution()


# ---------------------------------------------------------------------------
# Phase 1 — PINN (ODE + IC only)
# ---------------------------------------------------------------------------

class PIPNNs:
    """
    Parametric PINN for an n-DOF impact-damper chain.

    Network  : t  →  (x_1, ..., x_n)
    Loss     : beta_icx * L_ic  +  beta_fx * L_ode

    After training, call find_impact_times() to locate the first impact
    for every impactor via root-finding on the frozen network.

    Parameters
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
        self.lb = np.asarray(lb, dtype=np.float32).reshape(1, -1)
        self.ub = np.asarray(ub, dtype=np.float32).reshape(1, -1)

        self.t  = np.asarray(t,  dtype=np.float32).reshape(-1, 1)
        self.t0 = np.asarray(t0, dtype=np.float32).reshape(-1, 1)

        self.x0  = np.asarray(x0_total,  dtype=np.float32)   # (1, n_dof)
        self.xt0 = np.asarray(xt0_total, dtype=np.float32)   # (1, n_dof)

        # Impactor ICs — stored as (n_dof,) float arrays
        self.y0  = np.asarray(y0,  dtype=np.float64).flatten()   # (n_dof,)
        self.yt0 = np.asarray(yt0, dtype=np.float64).flatten()   # (n_dof,)

        self.n_dof = int(n_dof)

        # Impact gap: scalar → same for all DOFs; array → per-DOF
        D_arr = np.asarray(D, dtype=np.float64)
        self.D = float(D_arr) * np.ones(self.n_dof) if D_arr.ndim == 0 else D_arr.flatten()

        self.phi  = float(phi)
        self.phi1 = float(np.squeeze(phi1))
        self.phi2 = float(np.squeeze(phi2))

        self.beta_icx = float(hyp_ini_weight_loss[0])
        self.beta_fx  = float(hyp_ini_weight_loss[1])

        self.layers = layers

        # Each instance owns an isolated graph — prevents node accumulation across
        # segments that would otherwise slow down sess.run() calls progressively.
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.weights, self.biases = self._init_NN(layers)

            # TF matrices
            self.M_tf = tf.constant(M, dtype=tf.float32)
            self.K_tf = tf.constant(K, dtype=tf.float32)
            if C is None:
                self.C_tf = tf.constant(np.zeros_like(M), dtype=tf.float32)
            else:
                self.C_tf = tf.constant(C, dtype=tf.float32)

            # Placeholders
            self.t_tf   = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
            self.t0_tf  = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
            self.x0_tf  = tf.compat.v1.placeholder(tf.float32, shape=[None, self.n_dof])
            self.xt0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.n_dof])

            # Graph
            self.x_pred, self.xt_pred, self.xtt_pred = self._net_u(self.t_tf)
            self.x0_pred, self.xt0_pred, _            = self._net_u(self.t0_tf)
            self.fx_pred                               = self._net_f(self.t_tf)

            # Losses
            self.loss_icx = (
                tf.reduce_mean(tf.square(self.x0_pred  - self.x0_tf)) +
                tf.reduce_mean(tf.square(self.xt0_pred - self.xt0_tf))
            )
            self.loss_fx = tf.reduce_mean(tf.square(self.fx_pred))
            self.loss    = self.beta_icx * self.loss_icx + self.beta_fx * self.loss_fx

            # Optimisers
            self.optimizer_Adam  = tf.compat.v1.train.AdamOptimizer()
            self.train_op_Adam   = self.optimizer_Adam.minimize(self.loss)
            if optimizer_LB:
                self.optimizer_LB = tf.contrib.opt.ScipyOptimizerInterface(
                    self.loss,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 50000,
                        'maxfun':  50000,
                        'maxcor':  50,
                        'maxls':   50,
                        'ftol':    1.0 * np.finfo(float).eps,
                    },
                )

            # Logs
            self.loss_log     = []
            self.loss_icx_log = []
            self.loss_fx_log  = []

            self.sess = tf.compat.v1.Session(
                graph=self.graph,
                config=tf.compat.v1.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False,
                )
            )
            self.sess.run(tf.compat.v1.global_variables_initializer())

    # ------------------------------------------------------------------
    # Network
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
            tf.random.truncated_normal(size, stddev=stddev), dtype=tf.float32
        )

    def _neural_net(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]

    def _net_u(self, t):
        """t → x, x_t, x_tt  (each [N, n_dof])"""
        x    = self._neural_net(t)
        x_t  = tf.concat([tf.gradients(x[:, i], t)[0]   for i in range(self.n_dof)], axis=1)
        x_tt = tf.concat([tf.gradients(x_t[:, i], t)[0] for i in range(self.n_dof)], axis=1)
        return x, x_t, x_tt

    def _net_f(self, t):
        """ODE residual: M x_tt + C x_t + K x - F(t)  →  [N, n_dof]"""
        x, x_t, x_tt = self._net_u(t)

        # External forcing on the last DOF only
        F = tf.concat(
            [tf.zeros_like(t)] * (self.n_dof - 1) +
            [self.phi1 * tf.sin(self.phi2 * np.pi * (t + self.phi))],
            axis=1,
        )  # [N, n_dof]

        residual = (
            tf.transpose(tf.matmul(self.M_tf, x_tt, transpose_b=True)) +
            tf.transpose(tf.matmul(self.C_tf, x_t,  transpose_b=True)) +
            tf.transpose(tf.matmul(self.K_tf, x,    transpose_b=True)) -
            F
        )
        return residual

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _feed(self):
        return {
            self.t_tf:   self.t,
            self.t0_tf:  self.t0,
            self.x0_tf:  self.x0,
            self.xt0_tf: self.xt0,
        }

    def _callback(self, loss, loss_icx, loss_fx):
        self.loss_log.append(loss)
        self.loss_icx_log.append(loss_icx)
        self.loss_fx_log.append(loss_fx)
        print('Loss: %.5e   Loss_ic: %.5e   Loss_ode: %.5e'
              % (loss, loss_icx, loss_fx))

    def train(self, nIter=1000, optimizer_LB=True, print_every=100):
        """
        Phase 1: train NN on ODE + IC loss.
        After this, call find_impact_times() for Phase 2.
        """
        fd = self._feed()
        t0 = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, fd)
            if it % print_every == 0:
                lv  = self.sess.run(self.loss,     fd)
                lic = self.sess.run(self.loss_icx, fd)
                lfx = self.sess.run(self.loss_fx,  fd)
                self.loss_log.append(lv)
                self.loss_icx_log.append(lic)
                self.loss_fx_log.append(lfx)
                print('It %5d  Loss %.3e  Loss_ic %.3e  Loss_ode %.3e  %.2fs'
                      % (it, lv, lic, lfx, time.time() - t0))
                t0 = time.time()

        if optimizer_LB and hasattr(self, 'optimizer_LB'):
            self.optimizer_LB.minimize(
                self.sess, feed_dict=fd,
                fetches=[self.loss, self.loss_icx, self.loss_fx],
                loss_callback=self._callback,
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, t):
        """
        Forward pass at query times t (shape [N, 1] or scalar).

        Returns
        -------
        x, xt, xtt : each (N, n_dof)
        """
        t = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        fd = {self.t_tf: t}
        x    = self.sess.run(self.x_pred,   fd)
        xt   = self.sess.run(self.xt_pred,  fd)
        xtt  = self.sess.run(self.xtt_pred, fd)
        return x, xt, xtt


# ---------------------------------------------------------------------------
# Phase 2 — root-finding for impact times (all n DOFs)
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

    Parameters
    ----------
    model  : trained PIPNNs instance
    y0     : (n_dof,) array — impactor IC positions
    yt0    : (n_dof,) array — impactor IC velocities
    D      : float or (n_dof,) — impact gap(s)
    T_max  : float — segment time window
    n_scan : int   — number of coarse scan points
    tol    : float — Brent's method tolerance

    Returns
    -------
    t_impacts : (n_dof,) float array — impact time for each DOF
                (set to T_max when no impact is detected)
    hit       : (n_dof,) bool array  — True when a genuine impact was found
    """
    n = model.n_dof
    y0  = np.asarray(y0,  dtype=np.float64).flatten()
    yt0 = np.asarray(yt0, dtype=np.float64).flatten()
    D_arr = np.asarray(D, dtype=np.float64)
    D_vec = float(D_arr) * np.ones(n) if D_arr.ndim == 0 else D_arr.flatten()
    T_max = float(T_max)

    # Single forward pass over n_scan points
    t_scan = np.linspace(1e-4 * T_max, T_max, n_scan).reshape(-1, 1).astype(np.float32)
    x_scan, _, _ = model.predict(t_scan)   # (n_scan, n_dof)
    t_flat = t_scan.flatten().astype(np.float64)

    t_impacts = np.full(n, T_max)
    hit       = np.zeros(n, dtype=bool)

    for i in range(n):
        y_scan = y0[i] + yt0[i] * t_flat
        gap    = np.abs(x_scan[:, i] - y_scan) - D_vec[i]

        # First interval where gap was negative (masses separated)
        neg_idx = np.where(gap < 0)[0]
        if len(neg_idx) == 0:
            print(f"  DOF {i+1}: gap never closed — no impact in [0, {T_max:.3f}]")
            continue

        # First upward zero-crossing after the first negative region
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

    Parameters
    ----------
    model       : PIPNNs — trained model for the current segment
    t_impact    : float  — time of the first impact (min of t_impacts)
    x0_post     : (1, n_dof) — primary DOF displacements at t_impact
    xt0_post    : (1, n_dof) — primary DOF velocities at t_impact (to update)
    y0          : (n_dof,) — impactor positions at segment start
    yt0         : (n_dof,) — impactor velocities at segment start
    hit_dof     : int — 0-based index of the DOF that impacts first
    mx_list     : (n_dof,) — primary masses
    my_list     : (n_dof,) — impactor masses
    r           : float — coefficient of restitution
    A_inv_B     : (2, 2) — velocity-update matrix [see notebook]

    Returns
    -------
    x0_next     : (1, n_dof) — displacement ICs for next segment
    xt0_next    : (1, n_dof) — velocity ICs for next segment
    y0_next     : (n_dof,) — impactor position ICs for next segment
    yt0_next    : (n_dof,) — impactor velocity ICs for next segment
    """
    n = model.n_dof
    t = float(t_impact)

    x0_next  = x0_post.copy()
    xt0_next = xt0_post.copy()

    # Impactor positions / velocities at impact time (free-flight)
    y0_next  = y0  + yt0  * t      # (n_dof,) displacements at t*
    yt0_next = yt0.copy()          # velocities unchanged (to be updated below)

    # Apply velocity jump for the impacting DOF
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
    Newmark-beta implicit integrator.

    Parameters
    ----------
    M, C, K   : (n_dof, n_dof)
    F         : (n_dof, n_steps) — external force at each time step
    dt        : float — time step
    n_steps   : int   — number of steps (including initial)
    n_dof     : int
    x0, xt0   : (1, n_dof) or None — initial conditions (default: zeros)
    beta, gamma : Newmark parameters (default: constant-average acceleration)

    Returns
    -------
    x, xt, xtt : each (n_dof, n_steps)
    """
    x   = np.zeros((n_dof, n_steps))
    xt  = np.zeros((n_dof, n_steps))
    xtt = np.zeros((n_dof, n_steps))

    if x0  is not None: x[:,  0] = np.asarray(x0).flatten()
    if xt0 is not None: xt[:, 0] = np.asarray(xt0).flatten()

    xtt[:, 0] = np.linalg.solve(M, F[:, 0] - C @ xt[:, 0] - K @ x[:, 0])

    K_eff = M / (beta * dt**2) + gamma * C / (beta * dt) + K
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

