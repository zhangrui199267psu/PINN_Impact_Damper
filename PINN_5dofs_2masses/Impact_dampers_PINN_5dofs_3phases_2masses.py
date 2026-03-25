"""
Impact_dampers_PINN_5dofs_3phases_2masses.py
--------------------------------------------
Two-phase PINN for a 5-DOF chain with 2 internal impact dampers.

System
------
5 DOFs: x1, x2, x3, x4, x5  (chain, spring-coupled)
  - DOF 3 (index 2): primary mass mx3, internal impactor y3 (free-flight)
  - DOF 4 (index 3): primary mass mx4, internal impactor y4 (free-flight)
  - DOF 5 (index 4): primary/forcing mass, external harmonic excitation
  - DOF 1, 2: no internal mass

EOM between impacts:
    M x_tt + K x = F(t)
    y_i(t) = y0_i + yt0_i * t   (free flight for each impactor)

Impact condition (cell i):
    |x_i(t*) - y_i(t*)| = D

Two-phase strategy
------------------
Phase 1  — train1()
    Minimise: beta_icx * L_ic + beta_fx * L_ode
    Network weights learn the physical response for the full segment.
    No impact-time variable.  Network is frozen after this phase.

Phase 2  — find_impact_times()
    Root-find t* for DOF 3 and DOF 4 independently on the frozen network.
    Uses a coarse scan then Brent's method to pinpoint the first crossing of
        gap_i(t) = |x_i(t) - y_i(t)| - D  through zero (from below).
    The segment ends at t_seg = min(t*_3, t*_4).

Author: Rui Zhang
"""

import os
import time
import numpy as np
import tensorflow as tf
from scipy.optimize import brentq

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

# TF1-compatible execution
tf.compat.v1.disable_eager_execution()


# ---------------------------------------------------------------------------
# Phase 1 — PINN (ODE + IC only, no lambda)
# ---------------------------------------------------------------------------

class PIPNNs:
    """
    Parametric PINN for the 5-DOF impact-damper chain.

    Network  : t  →  (x1, x2, x3, x4, x5)
    Loss     : beta_icx * L_ic + beta_fx * L_ode

    No learnable impact time.  After training, call find_impact_times() to
    locate t* for DOF 3 and DOF 4 via root-finding on the frozen network.
    """

    def __init__(
        self,
        lb, ub,
        t0, t,
        x0_total, xt0_total,
        y03, yt03,
        y04, yt04,
        M, K, D,
        n_dof,
        phi, phi1, phi2,
        layers,
        hyp_ini_weight_loss,
        optimizer_LB=True,
    ):
        """
        Parameters
        ----------
        lb, ub          : (1,) arrays — time-domain bounds for normalisation
        t0, t           : (1,1) and (N,1) — IC time and collocation times
        x0_total        : (1, n_dof) — displacement ICs
        xt0_total       : (1, n_dof) — velocity ICs
        y03, yt03       : (1,1) — IC position/velocity for impactor of DOF 3
        y04, yt04       : (1,1) — IC position/velocity for impactor of DOF 4
        M, K            : (n_dof, n_dof) — mass and stiffness matrices
        D               : float — impact gap (same for both impactors)
        n_dof           : int — number of DOFs (5)
        phi             : (1,1) — phase offset (accumulated segment time)
        phi1, phi2      : (1,1) — forcing amplitude and frequency
        layers          : list[int] — network architecture, e.g. [1, 64, 5]
        hyp_ini_weight_loss : (3,) — weights (beta_icx, beta_fx, unused)
        optimizer_LB    : bool — run L-BFGS-B after Adam
        """
        self.lb = np.asarray(lb, dtype=np.float32).reshape(1, -1)
        self.ub = np.asarray(ub, dtype=np.float32).reshape(1, -1)

        self.t  = np.asarray(t,  dtype=np.float32).reshape(-1, 1)
        self.t0 = np.asarray(t0, dtype=np.float32).reshape(-1, 1)

        self.x0  = np.asarray(x0_total,  dtype=np.float32)   # (1, n_dof)
        self.xt0 = np.asarray(xt0_total, dtype=np.float32)   # (1, n_dof)

        self.y03  = float(y03)
        self.yt03 = float(yt03)
        self.y04  = float(y04)
        self.yt04 = float(yt04)

        self.D     = float(D)
        self.n_dof = int(n_dof)

        self.phi  = float(phi)
        self.phi1 = float(phi1) if np.ndim(phi1) == 0 else float(np.squeeze(phi1))
        self.phi2 = float(phi2) if np.ndim(phi2) == 0 else float(np.squeeze(phi2))

        self.beta_icx = float(hyp_ini_weight_loss[0])
        self.beta_fx  = float(hyp_ini_weight_loss[1])

        # Network
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Placeholders
        self.t_tf   = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.t0_tf  = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x0_tf  = tf.compat.v1.placeholder(tf.float32, shape=[None, self.n_dof])
        self.xt0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.n_dof])

        # TF constants for M and K
        self.M_tf = tf.constant(M, dtype=tf.float32)
        self.K_tf = tf.constant(K, dtype=tf.float32)

        # Graph
        self.x_pred, self.xt_pred, self.xtt_pred = self.net_u(self.t_tf)
        self.x0_pred, self.xt0_pred, _            = self.net_u(self.t0_tf)
        self.fx_pred                               = self.net_f(self.t_tf)

        # Losses
        self.loss_icx = (
            tf.reduce_mean(tf.square(self.x0_pred  - self.x0_tf)) +
            tf.reduce_mean(tf.square(self.xt0_pred - self.xt0_tf))
        )
        self.loss_fx = tf.reduce_mean(tf.square(self.fx_pred))
        self.loss    = self.beta_icx * self.loss_icx + self.beta_fx * self.loss_fx

        # Optimisers
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
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
        self.train_op_Adam  = self.optimizer_Adam.minimize(self.loss)

        # Logs
        self.loss_log     = []
        self.loss_icx_log = []
        self.loss_fx_log  = []

        # Session
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
            )
        )
        self.sess.run(tf.compat.v1.global_variables_initializer())

    # ------------------------------------------------------------------
    # Network utilities
    # ------------------------------------------------------------------
    def initialize_NN(self, layers):
        weights, biases = [], []
        for l in range(len(layers) - 1):
            W = self.xavier_init([layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        stddev = np.sqrt(2.0 / (size[0] + size[1]))
        return tf.Variable(
            tf.random.truncated_normal(size, stddev=stddev), dtype=tf.float32
        )

    def neural_net(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]

    def net_u(self, t):
        """t → x, x_t, x_tt  (each: [N, n_dof])"""
        x    = self.neural_net(t)
        x_t  = tf.concat([tf.gradients(x[:, i], t)[0] for i in range(self.n_dof)], axis=1)
        x_tt = tf.concat([tf.gradients(x_t[:, i], t)[0] for i in range(self.n_dof)], axis=1)
        return x, x_t, x_tt

    def net_f(self, t):
        """ODE residual: M x_tt + K x - F(t)  →  [N, n_dof]"""
        x, _, x_tt = self.net_u(t)

        # External forcing: only on the last DOF
        F = tf.concat(
            [tf.zeros_like(t)] * (self.n_dof - 1) +
            [self.phi1 * tf.sin(self.phi2 * np.pi * (t + self.phi))],
            axis=1,
        )  # shape: [N, n_dof]

        # M x_tt^T + K x^T - F^T  →  transpose back to [N, n_dof]
        residual = (
            tf.transpose(tf.matmul(self.M_tf, x_tt, transpose_b=True)) +
            tf.transpose(tf.matmul(self.K_tf, x,    transpose_b=True)) -
            F
        )
        return residual

    # ------------------------------------------------------------------
    # Phase 1 — training
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
        print('Loss: %.5e,  Loss_icx: %.5e,  Loss_fx: %.5e'
              % (loss, loss_icx, loss_fx))

    def train(self, nIter=1000, optimizer_LB=True, print_every=100):
        """
        Phase 1: train NN weights with ODE + IC loss only.
        After this, call find_impact_times() for Phase 2.
        """
        fd = self._feed()
        start = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, fd)
            if it % print_every == 0:
                loss_v    = self.sess.run(self.loss,     fd)
                loss_ic_v = self.sess.run(self.loss_icx, fd)
                loss_fx_v = self.sess.run(self.loss_fx,  fd)
                self.loss_log.append(loss_v)
                self.loss_icx_log.append(loss_ic_v)
                self.loss_fx_log.append(loss_fx_v)
                print('It: %d,  Loss: %.3e,  Loss_icx: %.3e,  Loss_fx: %.3e,  Time: %.2f'
                      % (it, loss_v, loss_ic_v, loss_fx_v, time.time() - start))
                start = time.time()

        if optimizer_LB and hasattr(self, 'optimizer_LB'):
            self.optimizer_LB.minimize(
                self.sess, feed_dict=fd,
                fetches=[self.loss, self.loss_icx, self.loss_fx],
                loss_callback=self._callback,
            )

    # keep old names for drop-in compatibility
    def train1(self, nIter=1000, optimizer_LB1=True, print_every=100):
        self.train(nIter=nIter, optimizer_LB=optimizer_LB1, print_every=print_every)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, t):
        """
        Forward pass at query times t (shape [N, 1]).

        Returns
        -------
        x_star, xt_star, xtt_star : each (N, n_dof)
        """
        t = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        fd = {self.t_tf: t}
        x_s   = self.sess.run(self.x_pred,   fd)
        xt_s  = self.sess.run(self.xt_pred,  fd)
        xtt_s = self.sess.run(self.xtt_pred, fd)
        return x_s, xt_s, xtt_s


# ---------------------------------------------------------------------------
# Phase 2 — root-finding for impact times (DOF 3 and DOF 4)
# ---------------------------------------------------------------------------

def find_impact_times(
    model,
    y03, yt03,
    y04, yt04,
    D,
    T_max,
    n_scan=500,
    tol=1e-8,
):
    """
    Phase 2: find the first impact time for DOF 3 and DOF 4 by root-finding
    on the frozen (already-trained) network.

    Strategy
    --------
    1. Evaluate the gap functions over n_scan points in (0, T_max]:
           gap_i(t) = |x_i(t) - y_i(t)| - D
       where y_i(t) = y0_i + yt0_i * t (free flight).
    2. For each impactor, detect the first upward zero-crossing
       (gap negative → positive, i.e. masses reunite after separation).
    3. Refine with Brent's method between the two bracketing scan points.

    Parameters
    ----------
    model       : trained PIPNNs instance
    y03, yt03   : float — IC position/velocity for DOF 3 impactor
    y04, yt04   : float — IC position/velocity for DOF 4 impactor
    D           : float — impact gap
    T_max       : float — segment time window
    n_scan      : int   — number of coarse scan points
    tol         : float — Brent tolerance

    Returns
    -------
    t_imp3      : float — impact time for DOF 3 (T_max if no impact)
    t_imp4      : float — impact time for DOF 4 (T_max if no impact)
    hit3        : bool  — True if a genuine impact was found for DOF 3
    hit4        : bool  — True if a genuine impact was found for DOF 4
    """
    y03, yt03 = float(y03), float(yt03)
    y04, yt04 = float(y04), float(yt04)
    D, T_max  = float(D),   float(T_max)

    # Coarse scan — skip t=0 (trivial solution |x0-y0|=D already satisfied)
    t_scan = np.linspace(1e-4 * T_max, T_max, n_scan).reshape(-1, 1).astype(np.float32)
    x_scan, _, _ = model.predict(t_scan)          # (n_scan, n_dof)
    t_flat        = t_scan.flatten()

    results = {}
    for dof_idx, (y0, yt0, label) in enumerate(
        [(y03, yt03, 3), (y04, yt04, 4)], start=0
    ):
        dof_col = 2 + dof_idx                     # column index in x (0-based: DOF3→2, DOF4→3)
        y_scan  = y0 + yt0 * t_flat
        gap     = np.abs(x_scan[:, dof_col] - y_scan) - D

        # Find first upward crossing (gap: negative → positive)
        sep_idx = np.where(gap < 0)[0]
        if len(sep_idx) == 0:
            print(f"  DOF {label}: gap never closed — no impact in [0, {T_max:.3f}]")
            results[label] = (T_max, False)
            continue

        gap_after = gap[sep_idx[0]:]
        up_cross  = np.where(np.diff(np.sign(gap_after)) > 0)[0]
        if len(up_cross) == 0:
            print(f"  DOF {label}: masses separated but never re-collided — no impact")
            results[label] = (T_max, False)
            continue

        bracket_idx = sep_idx[0] + up_cross[0]
        ta = float(t_flat[bracket_idx])
        tb = float(t_flat[bracket_idx + 1])

        def _gap(t_val, _y0=y0, _yt0=yt0, _col=dof_col):
            xv, _, _ = model.predict(np.array([[t_val]], dtype=np.float32))
            yv = _y0 + _yt0 * t_val
            return abs(float(xv[0, _col]) - yv) - D

        try:
            t_imp = brentq(_gap, ta, tb, xtol=tol)
        except ValueError:
            t_imp = ta

        print(f"  DOF {label}: impact at t = {t_imp:.6f}")
        results[label] = (t_imp, True)

    t_imp3, hit3 = results[3]
    t_imp4, hit4 = results[4]
    return t_imp3, t_imp4, hit3, hit4


# ---------------------------------------------------------------------------
# Velocity update at impact
# ---------------------------------------------------------------------------

def impact_velocity_update(mx, my, r, xt_minus, yt_minus):
    """
    Post-impact velocities via momentum conservation + restitution coefficient.

    [mx  my][xt+]   [mx - r*my   my*(1+r) ][xt-]
    [ 1  -1][yt+] = [mx*(1+r)    my - r*mx][yt-]  / (mx+my)

    Parameters — all floats.
    """
    mx, my = float(mx), float(my)
    xt_m, yt_m = float(xt_minus), float(yt_minus)
    total  = mx + my
    xt_p   = ((mx - r * my) * xt_m + my * (1.0 + r) * yt_m) / total
    yt_p   = (mx * (1.0 + r) * xt_m + (my - r * mx) * yt_m) / total
    return xt_p, yt_p
