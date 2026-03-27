"""
pinn_ndof_chain_parametric_tf2.py
----------------------------------
Parametric PINN for an n-DOF impact-damper chain (TensorFlow 2).

One network handles multiple parameter cases simultaneously, so a single
training run covers the full parameter space.  After training, predictions
for any parameter combination within the sampled range are free.

Network
-------
    f(t, mx, my, k, phi1, phi2)  →  (x_1, …, x_{n_dof})

Varied physical parameters (one set per training case):
    mx    — primary (outer) mass
    my    — impactor (inner) mass  [only enters impact update, not ODE]
    k     — nearest-neighbour coupling spring stiffness
    phi1  — forcing amplitude
    phi2  — forcing angular-frequency coefficient  [F = phi1·sin(phi2·π·(t+φ))]

Fixed (set at construction):
    n_dof — number of DOFs / impactors
    c     — proportional damping coefficient  (C = c·K_unit)
    D     — impact gap (uniform across DOFs)
    r     — coefficient of restitution

Physics between impacts
-----------------------
    mx · ẍ  +  c · K_unit·ẋ  +  k · K_unit·x  =  F(t)

where K_unit is the normalised tridiagonal coupling matrix (k=1):
    (K_unit)_ii = 2 for interior DOFs,  (K_unit)_ii = 1 at boundary DOFs
    (K_unit)_{i,i±1} = -1

    and F_i = 0 for i < n_dof-1,  F_{n_dof-1} = phi1·sin(phi2·π·(t+φ))

Public API
----------
    lhs_sample(n_cases, lb, ub)          — Latin-Hypercube parameter sampling
    ParametricPIPNNs                      — core parametric PINN class
    find_impact_times_parametric(model, j)— root-find first impact per DOF, case j
    impact_velocity_update(mx,my,r,xt,yt) — momentum + restitution
    propagate_ics_parametric(...)         — compute post-impact ICs for case j
    newmark_beta(...)                     — Newmark-β reference integrator

Author: Rui Zhang
"""

import os
import time
import numpy as np
import tensorflow as tf
from scipy.optimize import brentq, minimize
from scipy.stats import qmc

# ── GPU / CPU selection ────────────────────────────────────────────────────
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # -1 = CPU, 0 = GPU0
np.random.seed(1234)
tf.random.set_seed(1234)


# ---------------------------------------------------------------------------
# Adam helper  (Keras 2 / Keras 3 compatible)
# ---------------------------------------------------------------------------
def _make_adam(lr=1e-3):
    """Return a working Adam optimiser regardless of Keras version."""
    for factory in [
        lambda: tf.keras.optimizers.Adam(learning_rate=lr),
        lambda: tf.keras.optimizers.legacy.Adam(learning_rate=lr),
    ]:
        try:
            return factory()
        except (AttributeError, ImportError):
            pass
    raise RuntimeError('No compatible tf.keras Adam optimiser found.')


# ---------------------------------------------------------------------------
# Latin-Hypercube sampling
# ---------------------------------------------------------------------------
def lhs_sample(n_cases, lb, ub, seed=42):
    """
    Draw n_cases samples from a Latin Hypercube over [lb, ub].

    Parameters
    ----------
    n_cases : int
    lb, ub  : array-like of length n_params

    Returns
    -------
    samples : (n_cases, n_params) float64 array
    """
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    sampler = qmc.LatinHypercube(d=len(lb), seed=seed)
    return qmc.scale(sampler.random(n=n_cases), lb, ub)


# ---------------------------------------------------------------------------
# Tridiagonal stiffness-matrix-vector product  (vectorised, per-row k)
# ---------------------------------------------------------------------------
def _tridiag_matvec(x, coeff):
    """
    Compute  coeff * K_unit @ x  per row, where K_unit is the normalised
    tridiagonal matrix for a 1-D chain with free ends:

        (K_unit @ x)_i = -x_{i-1} + 2·x_i - x_{i+1}
    with ghost-cell boundary conditions: x_{-1} = x_0, x_n = x_{n-1}.

    Parameters
    ----------
    x     : tf.Tensor  [N, n_dof]
    coeff : tf.Tensor  [N, 1]     — per-row scalar (k_col or c_col)

    Returns
    -------
    tf.Tensor  [N, n_dof]
    """
    x_pad = tf.concat([x[:, :1], x, x[:, -1:]], axis=1)   # [N, n_dof+2]
    return coeff * (-x_pad[:, :-2] + 2.0 * x - x_pad[:, 2:])


# ---------------------------------------------------------------------------
# Parametric PINN class
# ---------------------------------------------------------------------------
class ParametricPIPNNs:
    """
    Parametric PINN for an n-DOF impact-damper chain (TF2).

    One network f(t, mx, my, k, phi1, phi2) → (x_1,…,x_n) is trained
    simultaneously on n_cases LHS-sampled parameter sets.

    Parameters
    ----------
    n_dof        : int
    params_cases : (n_cases, 5) — [mx, my, k, phi1, phi2] per case
    lb_params    : (5,) lower normalisation bounds for the 5 parameters
    ub_params    : (5,) upper normalisation bounds
    c_damp       : float — fixed proportional damping coefficient (0 = undamped)
    T_seg        : float — segment duration [0, T_seg]
    N_col        : int   — collocation points per case
    phi_offsets  : (n_cases,) — accumulated phase offset φ per case
    x0_cases     : (n_cases, n_dof) — IC displacements
    xt0_cases    : (n_cases, n_dof) — IC velocities
    y0_cases     : (n_cases, n_dof) — impactor IC positions
    yt0_cases    : (n_cases, n_dof) — impactor IC velocities
    D            : float — impact gap (uniform across DOFs and cases)
    layers       : list[int] e.g. [6, 64, 64, n_dof]
    hyp_ini_weight_loss : [beta_ic, beta_ode]
    optimizer_LB : bool — run L-BFGS-B polishing after Adam
    """

    N_PARAMS = 5   # mx, my, k, phi1, phi2
    INPUT_DIM = 6  # 1 (time) + 5 (params)

    def __init__(
        self,
        n_dof,
        params_cases,
        lb_params,
        ub_params,
        c_damp,
        T_seg,
        N_col,
        phi_offsets,
        x0_cases,
        xt0_cases,
        y0_cases,
        yt0_cases,
        D,
        layers,
        hyp_ini_weight_loss,
        optimizer_LB=True,
    ):
        self.n_dof   = int(n_dof)
        self.n_cases = int(params_cases.shape[0])
        self.T_seg   = float(T_seg)
        self.D       = float(D)
        self.c_damp  = float(c_damp)

        # ── normalisation bounds: [t_lb, param_lb...] ─────────────────────
        lb_all = np.array([0.0, *lb_params], dtype=np.float32)
        ub_all = np.array([T_seg, *ub_params], dtype=np.float32)
        self.lb = tf.constant(lb_all.reshape(1, -1))   # [1, 6]
        self.ub = tf.constant(ub_all.reshape(1, -1))   # [1, 6]

        # ── params per case ───────────────────────────────────────────────
        p = np.asarray(params_cases, dtype=np.float32)  # [n_cases, 5]
        self.params_cases = tf.constant(p)

        # ── accumulated phase offsets ──────────────────────────────────────
        self.phi_offsets_np = np.asarray(phi_offsets, dtype=np.float32).reshape(-1, 1)

        # ── IC arrays (numpy for external use; tf.constant for loss) ──────
        self.x0_cases  = np.asarray(x0_cases,  dtype=np.float32)
        self.xt0_cases = np.asarray(xt0_cases, dtype=np.float32)
        self.y0_cases  = np.asarray(y0_cases,  dtype=np.float32)
        self.yt0_cases = np.asarray(yt0_cases, dtype=np.float32)
        self.x0_tf     = tf.constant(self.x0_cases)    # [n_cases, n_dof]
        self.xt0_tf    = tf.constant(self.xt0_cases)

        # ── collocation data (ODE loss) ────────────────────────────────────
        # t: N_col uniform points in (0, T_seg], tiled for each case
        t_one = np.linspace(0, T_seg, N_col, dtype=np.float32).reshape(-1, 1)
        t_rep = np.tile(t_one, (self.n_cases, 1))                 # [n*N, 1]
        p_rep = np.repeat(p, N_col, axis=0)                       # [n*N, 5]
        phi_rep = np.repeat(self.phi_offsets_np, N_col, axis=0)   # [n*N, 1]

        self.t_col      = tf.constant(t_rep)     # [n_cases*N_col, 1]
        self.params_col = tf.constant(p_rep)     # [n_cases*N_col, 5]
        self.phi_col    = tf.constant(phi_rep)   # [n_cases*N_col, 1]

        # ── IC collocation data (IC loss) ──────────────────────────────────
        t0_all  = np.zeros((self.n_cases, 1), dtype=np.float32)
        self.t0_col      = tf.constant(t0_all)
        self.params0_col = self.params_cases       # [n_cases, 5]
        self.phi0_col    = tf.constant(self.phi_offsets_np)

        # ── loss weights ───────────────────────────────────────────────────
        self.beta_icx = float(hyp_ini_weight_loss[0])
        self.beta_fx  = float(hyp_ini_weight_loss[1])

        # ── network ────────────────────────────────────────────────────────
        self.layers = layers
        self.weights, self.biases = self._init_NN(layers)
        self.trainable_vars = self.weights + self.biases

        # ── optimiser ──────────────────────────────────────────────────────
        self.adam = _make_adam(lr=1e-3)
        self._use_lbfgs = optimizer_LB

        # ── logs ───────────────────────────────────────────────────────────
        self.loss_log     = []
        self.loss_icx_log = []
        self.loss_fx_log  = []

    # ------------------------------------------------------------------
    # Network initialisation
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

    def _neural_net(self, inp):
        """
        inp : [N, 6]  — [t, mx, my, k, phi1, phi2] (normalised internally)
        Returns [N, n_dof]
        """
        H = 2.0 * (inp - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]

    # ------------------------------------------------------------------
    # Time derivatives  (gradient w.r.t. t only)
    # ------------------------------------------------------------------
    def _net_u(self, t_col, params_col):
        """
        Compute x, ẋ, ẍ via nested GradientTape + batch_jacobian.

        t_col     : [N, 1]  — watched tensor
        params_col: [N, 5]

        Returns x [N,n], x_t [N,n], x_tt [N,n]
        """
        with tf.GradientTape() as tape2:
            tape2.watch(t_col)
            with tf.GradientTape() as tape1:
                tape1.watch(t_col)
                inp = tf.concat([t_col, params_col], axis=1)  # [N, 6]
                x = self._neural_net(inp)                      # [N, n_dof]
            x_t  = tf.squeeze(tape1.batch_jacobian(x,   t_col), axis=-1)  # [N, n_dof]
        x_tt = tf.squeeze(tape2.batch_jacobian(x_t, t_col), axis=-1)      # [N, n_dof]
        return x, x_t, x_tt

    # ------------------------------------------------------------------
    # ODE residual
    # ------------------------------------------------------------------
    def _net_f(self, t_col, params_col, phi_col):
        """
        mx·ẍ + c·K_unit·ẋ + k·K_unit·x  −  F(t)

        Parameters extracted from params_col:
            col 0: mx   col 2: k   col 3: phi1   col 4: phi2
            col 1: my   (not in ODE — only used in impact update)
        """
        x, x_t, x_tt = self._net_u(t_col, params_col)

        mx_col   = params_col[:, 0:1]   # [N, 1]
        k_col    = params_col[:, 2:3]   # [N, 1]
        phi1_col = params_col[:, 3:4]   # [N, 1]
        phi2_col = params_col[:, 4:5]   # [N, 1]

        # K·x and C·ẋ  (tridiagonal, per-row spring constant)
        Kx  = _tridiag_matvec(x,   k_col)                          # [N, n_dof]
        Cxt = _tridiag_matvec(x_t, tf.ones_like(k_col) * self.c_damp)  # [N, n_dof]

        # External force: last DOF only
        F_last = phi1_col * tf.sin(phi2_col * np.pi * (t_col + phi_col))
        F = tf.concat(
            [tf.zeros_like(x[:, :-1]), F_last],
            axis=1,
        )  # [N, n_dof]

        return mx_col * x_tt + Cxt + Kx - F   # [N, n_dof]

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _compute_loss(self):
        # IC loss
        x0_pred, xt0_pred, _ = self._net_u(self.t0_col, self.params0_col)
        loss_icx = (
            tf.reduce_mean(tf.square(x0_pred  - self.x0_tf)) +
            tf.reduce_mean(tf.square(xt0_pred - self.xt0_tf))
        )
        # ODE residual loss
        res     = self._net_f(self.t_col, self.params_col, self.phi_col)
        loss_fx = tf.reduce_mean(tf.square(res))

        loss = self.beta_icx * loss_icx + self.beta_fx * loss_fx
        return loss, loss_icx, loss_fx

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @tf.function
    def _train_step_adam(self):
        with tf.GradientTape() as tape:
            loss, loss_icx, loss_fx = self._compute_loss()
        grads = tape.gradient(loss, self.trainable_vars)
        self.adam.apply_gradients(zip(grads, self.trainable_vars))
        return loss, loss_icx, loss_fx

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
        with tf.GradientTape() as tape:
            loss, loss_icx, loss_fx = self._compute_loss()
        grads = tape.gradient(loss, self.trainable_vars)
        return loss, loss_icx, loss_fx, grads

    def _lbfgs_objective(self, flat_params_f64):
        self._set_flat_params(flat_params_f64.astype(np.float32))
        loss, loss_icx, loss_fx, grads = self._loss_and_grads_tf()
        lv, lic, lfx = float(loss), float(loss_icx), float(loss_fx)
        self.loss_log.append(lv)
        self.loss_icx_log.append(lic)
        self.loss_fx_log.append(lfx)
        print('Loss: %.5e   Loss_ic: %.5e   Loss_ode: %.5e' % (lv, lic, lfx))
        grad_flat = np.concatenate(
            [g.numpy().flatten() for g in grads]
        ).astype(np.float64)
        return lv, grad_flat

    def train(self, nIter=1000, optimizer_LB=True, print_every=100):
        """Adam + optional L-BFGS-B polishing."""
        t0_wall = time.time()
        for it in range(nIter):
            loss, loss_icx, loss_fx = self._train_step_adam()
            if it % print_every == 0:
                lv, lic, lfx = float(loss), float(loss_icx), float(loss_fx)
                self.loss_log.append(lv)
                self.loss_icx_log.append(lic)
                self.loss_fx_log.append(lfx)
                print('It %5d  Loss %.3e  Loss_ic %.3e  Loss_ode %.3e  %.2fs'
                      % (it, lv, lic, lfx, time.time() - t0_wall))
                t0_wall = time.time()

        if optimizer_LB and self._use_lbfgs:
            minimize(
                self._lbfgs_objective,
                self._get_flat_params().astype(np.float64),
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
    def predict(self, t_vals, case_idx):
        """
        Predict x, ẋ, ẍ for a single parameter case.

        Parameters
        ----------
        t_vals   : (N,) or (N,1) — times in [0, T_seg]
        case_idx : int — which of the n_cases to predict

        Returns
        -------
        x, x_t, x_tt : each (N, n_dof) numpy
        """
        t_arr = np.asarray(t_vals, dtype=np.float32).reshape(-1, 1)
        N = t_arr.shape[0]
        t_tf = tf.constant(t_arr)
        p_j  = tf.repeat(self.params_cases[case_idx:case_idx + 1, :], N, axis=0)
        x, x_t, x_tt = self._net_u(t_tf, p_j)
        return x.numpy(), x_t.numpy(), x_tt.numpy()

    def predict_all_cases(self, t_vals):
        """
        Predict for all n_cases at the same time points.

        Returns
        -------
        x_all : (n_cases, N, n_dof) numpy
        """
        return np.stack(
            [self.predict(t_vals, j)[0] for j in range(self.n_cases)],
            axis=0,
        )


# ---------------------------------------------------------------------------
# Phase 2 — impact-time root-finding (per case)
# ---------------------------------------------------------------------------

def find_impact_times_parametric(model, case_idx, n_scan=500, tol=1e-8):
    """
    Find the first impact time for every DOF in case case_idx.

    Strategy: scan the gap function, detect sign crossings, refine with Brent.

    Returns
    -------
    t_impacts : (n_dof,) float — first impact time per DOF (T_seg if none found)
    hit       : (n_dof,) bool
    """
    n     = model.n_dof
    T_max = model.T_seg
    y0    = model.y0_cases[case_idx].astype(np.float64)
    yt0   = model.yt0_cases[case_idx].astype(np.float64)
    D     = model.D

    t_scan = np.linspace(1e-4 * T_max, T_max, n_scan, dtype=np.float32)
    x_scan, _, _ = model.predict(t_scan, case_idx)
    t_flat = t_scan.astype(np.float64)

    t_impacts = np.full(n, T_max)
    hit       = np.zeros(n, dtype=bool)

    for i in range(n):
        y_scan = y0[i] + yt0[i] * t_flat
        gap    = np.abs(x_scan[:, i] - y_scan) - D

        neg_idx = np.where(gap < 0)[0]
        if not len(neg_idx):
            print(f'  Case {case_idx}, DOF {i+1}: no impact in [0, {T_max:.3f}]')
            continue

        gap_tail = gap[neg_idx[0]:]
        up_cross = np.where(np.diff(np.sign(gap_tail)) > 0)[0]
        if not len(up_cross):
            print(f'  Case {case_idx}, DOF {i+1}: separated but never re-collided')
            continue

        bracket_i = neg_idx[0] + up_cross[0]
        ta = float(t_flat[bracket_i])
        tb = float(t_flat[bracket_i + 1])

        def _gap(t_val, _i=i, _y0=y0[i], _yt0=yt0[i]):
            xv, _, _ = model.predict(np.array([t_val], dtype=np.float32), case_idx)
            return abs(float(xv[0, _i]) - (_y0 + _yt0 * t_val)) - D

        try:
            t_imp = brentq(_gap, ta, tb, xtol=tol)
        except ValueError:
            t_imp = ta

        print(f'  Case {case_idx}, DOF {i+1}: impact at t = {t_imp:.6f}')
        t_impacts[i] = t_imp
        hit[i]       = True

    return t_impacts, hit


# ---------------------------------------------------------------------------
# Impact velocity update
# ---------------------------------------------------------------------------

def impact_velocity_update(mx, my, r, xt_minus, yt_minus):
    """
    Post-impact velocities via momentum conservation + restitution.
    Returns (xt_plus, yt_plus) as floats.
    """
    mx, my = float(mx), float(my)
    r      = float(r)
    xt_m, yt_m = float(xt_minus), float(yt_minus)
    total  = mx + my
    xt_p   = ((mx - r * my) * xt_m + my * (1.0 + r) * yt_m) / total
    yt_p   = (mx * (1.0 + r) * xt_m + (my - r * mx) * yt_m) / total
    return xt_p, yt_p


# ---------------------------------------------------------------------------
# IC propagation (per case)
# ---------------------------------------------------------------------------

def propagate_ics_parametric(model, case_idx, t_impact, first_dof, r,
                              x0_cur, xt0_cur, y0_cur, yt0_cur):
    """
    Compute post-impact initial conditions for case case_idx.

    Returns
    -------
    x0_next, xt0_next : (n_dof,) — updated primary-mass ICs
    y0_next, yt0_next : (n_dof,) — updated impactor ICs
    """
    t  = float(t_impact)
    mx = float(model.params_cases[case_idx, 0])
    my = float(model.params_cases[case_idx, 1])

    x0_next  = x0_cur.copy()
    xt0_next = xt0_cur.copy()
    y0_next  = y0_cur  + yt0_cur * t
    yt0_next = yt0_cur.copy()

    xt_p, yt_p = impact_velocity_update(
        mx, my, r,
        xt0_cur[first_dof],
        yt0_cur[first_dof],
    )
    xt0_next[first_dof] = xt_p
    yt0_next[first_dof] = yt_p

    return x0_next, xt0_next, y0_next, yt0_next


# ---------------------------------------------------------------------------
# Newmark-beta reference integrator  (identical to non-parametric version)
# ---------------------------------------------------------------------------

def newmark_beta(M, C, K, F, dt, n_steps, n_dof, x0=None, xt0=None,
                 beta=0.25, gamma=0.5):
    """Standard Newmark-β implicit integrator."""
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
                   + xtt[:, i-1] * (0.5 / beta - 1))
            + C @ (gamma * x[:, i-1] / (beta * dt)
                   - xt[:, i-1] * (1 - gamma / beta)
                   - dt * xtt[:, i-1] * (1 - gamma / (2 * beta)))
        )
        x[:, i]   = K_eff_inv @ F_eff
        xtt[:, i] = ((x[:, i] - x[:, i-1]) / (beta * dt**2)
                     - xt[:, i-1] / (beta * dt)
                     - xtt[:, i-1] * (0.5 / beta - 1))
        xt[:, i]  = (xt[:, i-1]
                     + dt * ((1 - gamma) * xtt[:, i-1] + gamma * xtt[:, i]))

    return x, xt, xtt
