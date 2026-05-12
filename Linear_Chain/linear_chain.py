"""
linear_chain.py
---------------
Solves a free-free linear chain (no internal impact-damper masses) with two
methods and compares their responses:

  1. Newmark-beta  — implicit reference integrator
  2. PINN          — single neural network trained on the full [0, T_END]
                     interval (no time-marching segmentation)

System:  M x_tt + C x_t + K x = 0
  - N_DOF = 20, free-free nearest-neighbour chain
  - Left-end initial velocity excitation: x_dot_1(0) = v0, all else zero
  - Three input-energy cases: low / medium / high

All output figures and data are saved to  Results_Linear_Chain/.

Run:
    python linear_chain.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.optimize import minimize

try:
    from scipy.io import savemat
    _HAS_SAVEMAT = True
except ImportError:
    _HAS_SAVEMAT = False

# ── device / reproducibility ───────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # set to '-1' to force CPU
np.random.seed(1234)
tf.random.set_seed(1234)

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════════

N_DOF  = 20
M_VAL  = 1.0    # mass (kg)
K_VAL  = 1.0    # spring stiffness (N/m)
C_VAL  = 0.0    # damping (undamped)

T_END  = 10.0   # total simulation time (s)

# Newmark-beta
DT_NM  = 0.001
BETA_NM  = 0.25
GAMMA_NM = 0.5

# PINN — single network on [0, T_END]
LAYERS      = [1, 128, 128, N_DOF]   # deeper/wider to handle 10 s
N_COLLOC    = 2000                    # collocation points
N_ITER_ADAM = 5000                    # Adam iterations
USE_LBFGS   = True

WEIGHT_IC  = 10.0   # higher IC weight keeps the network anchored at t=0
WEIGHT_ODE = 1.0

INPUT_VELOCITY_CASES = {
    'low':    -1.0,
    'medium': -2.0,
    'high':   -10.0,
}
SELECTED_DOFS = [0, 4, 9, 14, 19]   # 0-indexed, for comparison plots
SAVE_DIR = 'Results_Linear_Chain'


# ═══════════════════════════════════════════════════════════════════════════════
# Keras-version-safe Adam
# ═══════════════════════════════════════════════════════════════════════════════

def _make_adam(lr=1e-3):
    for factory in [
        lambda: tf.keras.optimizers.Adam(learning_rate=lr),
        lambda: tf.keras.optimizers.legacy.Adam(learning_rate=lr),
    ]:
        try:
            return factory()
        except (AttributeError, ImportError):
            pass
    raise RuntimeError("No compatible tf.keras Adam optimiser found.")


# ═══════════════════════════════════════════════════════════════════════════════
# System helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_matrices(n=N_DOF, m=M_VAL, k=K_VAL, c=C_VAL):
    """Build M, C, K for a free-free nearest-neighbour chain."""
    M = m * np.eye(n)
    K = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            K[i, i] += k;  K[i, i-1] -= k
            C[i, i] += c;  C[i, i-1] -= c
        if i < n - 1:
            K[i, i] += k;  K[i, i+1] -= k
            C[i, i] += c;  C[i, i+1] -= c
    return M, C, K


def left_velocity_ic(n=N_DOF, v0=1.0):
    """x_i(0)=0,  x_dot_1(0)=v0,  rest zero."""
    x0  = np.zeros(n)
    xt0 = np.zeros(n)
    xt0[0] = float(v0)
    return x0, xt0


# ═══════════════════════════════════════════════════════════════════════════════
# Newmark-beta integrator
# ═══════════════════════════════════════════════════════════════════════════════

def newmark_beta(M, C, K, x0, xt0,
                 t_end=T_END, dt=DT_NM,
                 beta=BETA_NM, gamma=GAMMA_NM):
    """
    Implicit Newmark-beta integrator for free vibration (F=0).

    Returns
    -------
    t  : (n_steps,)
    x  : (n_steps, n_dof)
    xt : (n_steps, n_dof)
    """
    n       = M.shape[0]
    n_steps = int(round(t_end / dt)) + 1
    t       = np.linspace(0.0, t_end, n_steps)

    xh   = np.zeros((n, n_steps))
    xth  = np.zeros((n, n_steps))
    xtth = np.zeros((n, n_steps))

    xh[:,  0] = x0
    xth[:, 0] = xt0
    F0 = -C @ xt0 - K @ x0
    xtth[:, 0] = np.linalg.solve(M, F0)

    K_eff = M / (beta*dt**2) + gamma*C / (beta*dt) + K
    K_inv = np.linalg.inv(K_eff)

    for i in range(1, n_steps):
        F_eff = (
            M @ (xh[:, i-1]/(beta*dt**2) + xth[:, i-1]/(beta*dt)
                 + xtth[:, i-1]*(0.5/beta - 1.0))
            + C @ (gamma*xh[:, i-1]/(beta*dt)
                   - xth[:, i-1]*(1.0 - gamma/beta)
                   - dt*xtth[:, i-1]*(1.0 - gamma/(2.0*beta)))
        )
        xh[:, i]   = K_inv @ F_eff
        xtth[:, i] = ((xh[:, i] - xh[:, i-1])/(beta*dt**2)
                      - xth[:, i-1]/(beta*dt)
                      - xtth[:, i-1]*(0.5/beta - 1.0))
        xth[:, i]  = (xth[:, i-1]
                      + dt*((1.0 - gamma)*xtth[:, i-1] + gamma*xtth[:, i]))

    return t, xh.T, xth.T   # (n_steps, n_dof)


# ═══════════════════════════════════════════════════════════════════════════════
# PINN — single network on [0, T_END]
# ═══════════════════════════════════════════════════════════════════════════════

class PINNLinearChain:
    """
    Single neural network trained on the full time interval [0, T_END].

    Network:  t  →  (x_1, ..., x_n)
    Loss:     w_ic * L_ic  +  w_ode * L_ode

    No time-marching; one training pass covers the entire simulation horizon.
    """

    def __init__(self, t_end, x0, xt0, M, C, K,
                 layers=LAYERS,
                 n_colloc=N_COLLOC,
                 w_ic=WEIGHT_IC, w_ode=WEIGHT_ODE,
                 use_lbfgs=USE_LBFGS):

        self.t_end    = float(t_end)
        self.n_dof    = M.shape[0]
        self.w_ic     = float(w_ic)
        self.w_ode    = float(w_ode)
        self.use_lbfgs = bool(use_lbfgs)

        # normalisation: map [0, T_END] → [-1, 1]
        self.lb = tf.constant([[0.0]], dtype=tf.float32)
        self.ub = tf.constant([[self.t_end]], dtype=tf.float32)

        # collocation points (uniform over [0, T_END])
        t_col = np.linspace(0.0, self.t_end, n_colloc).reshape(-1, 1).astype(np.float32)
        self.t_col = tf.constant(t_col)

        # IC data (t=0)
        self.t0_data  = tf.constant([[0.0]], dtype=tf.float32)
        self.x0_data  = tf.constant(x0.reshape(1, -1),  dtype=tf.float32)
        self.xt0_data = tf.constant(xt0.reshape(1, -1), dtype=tf.float32)

        # system matrices
        self.M_tf = tf.constant(M, dtype=tf.float32)
        self.C_tf = tf.constant(C, dtype=tf.float32)
        self.K_tf = tf.constant(K, dtype=tf.float32)

        # network
        self.layers = layers
        self.weights, self.biases = self._init_nn(layers)
        self.trainable_vars = self.weights + self.biases

        self.adam     = _make_adam(lr=1e-3)
        self.loss_log = []

    # ── network ────────────────────────────────────────────────────────────────
    def _init_nn(self, layers):
        ws, bs = [], []
        for l in range(len(layers) - 1):
            stddev = np.sqrt(2.0 / (layers[l] + layers[l+1]))
            W = tf.Variable(tf.random.truncated_normal(
                [layers[l], layers[l+1]], stddev=stddev, dtype=tf.float32
            ))
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32))
            ws.append(W); bs.append(b)
        return ws, bs

    def _net(self, t):
        H = 2.0 * (t - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]

    # ── derivatives via nested GradientTape ───────────────────────────────────
    def _net_u(self, t):
        """Returns x, x_t, x_tt — each (N, n_dof)."""
        with tf.GradientTape() as t2:
            t2.watch(t)
            with tf.GradientTape() as t1:
                t1.watch(t)
                x   = self._net(t)
            x_t  = tf.squeeze(t1.batch_jacobian(x,   t), axis=-1)
        x_tt = tf.squeeze(t2.batch_jacobian(x_t, t), axis=-1)
        return x, x_t, x_tt

    # ── loss ──────────────────────────────────────────────────────────────────
    def _compute_loss(self):
        # ODE residual at collocation points
        x, x_t, x_tt = self._net_u(self.t_col)
        res = (
            tf.transpose(tf.matmul(self.M_tf, x_tt, transpose_b=True))
            + tf.transpose(tf.matmul(self.C_tf, x_t,  transpose_b=True))
            + tf.transpose(tf.matmul(self.K_tf, x,    transpose_b=True))
        )
        loss_ode = tf.reduce_mean(tf.square(res))

        # IC loss at t=0
        x0_pred, xt0_pred, _ = self._net_u(self.t0_data)
        loss_ic = (tf.reduce_mean(tf.square(x0_pred  - self.x0_data))
                   + tf.reduce_mean(tf.square(xt0_pred - self.xt0_data)))

        return self.w_ic * loss_ic + self.w_ode * loss_ode, loss_ic, loss_ode

    # ── Adam step ─────────────────────────────────────────────────────────────
    @tf.function
    def _adam_step(self):
        with tf.GradientTape() as tape:
            loss, l_ic, l_ode = self._compute_loss()
        grads = tape.gradient(loss, self.trainable_vars)
        self.adam.apply_gradients(zip(grads, self.trainable_vars))
        return loss, l_ic, l_ode

    # ── L-BFGS-B helpers ──────────────────────────────────────────────────────
    def _get_flat(self):
        return np.concatenate([v.numpy().flatten() for v in self.trainable_vars])

    def _set_flat(self, flat_f32):
        idx = 0
        for v in self.trainable_vars:
            n = int(np.prod(v.shape))
            v.assign(tf.reshape(flat_f32[idx:idx+n], v.shape))
            idx += n

    @tf.function
    def _loss_grads(self):
        with tf.GradientTape() as tape:
            loss, l_ic, l_ode = self._compute_loss()
        grads = tape.gradient(loss, self.trainable_vars)
        return loss, l_ic, l_ode, grads

    def _lbfgs_obj(self, flat_f64):
        self._set_flat(flat_f64.astype(np.float32))
        loss, l_ic, l_ode, grads = self._loss_grads()
        lv = float(loss)
        self.loss_log.append(lv)
        print(f'  L-BFGS  total {lv:.4e}  IC {float(l_ic):.4e}'
              f'  ODE {float(l_ode):.4e}')
        grad_flat = np.concatenate(
            [g.numpy().flatten() for g in grads]
        ).astype(np.float64)
        return lv, grad_flat

    # ── train ─────────────────────────────────────────────────────────────────
    def train(self, n_iter=N_ITER_ADAM, print_every=500):
        t_wall = time.time()
        for it in range(n_iter):
            loss, l_ic, l_ode = self._adam_step()
            if it % print_every == 0:
                lv = float(loss)
                self.loss_log.append(lv)
                print(f'  Adam {it:6d}  total {lv:.4e}'
                      f'  IC {float(l_ic):.4e}  ODE {float(l_ode):.4e}'
                      f'  ({time.time()-t_wall:.1f}s)')
                t_wall = time.time()

        if self.use_lbfgs:
            print('\n  Starting L-BFGS-B polishing...')
            minimize(
                self._lbfgs_obj,
                self._get_flat().astype(np.float64),
                method='L-BFGS-B', jac=True,
                options={
                    'maxiter': 50000, 'maxfun': 50000,
                    'maxcor': 50,    'maxls':  50,
                    'ftol': np.finfo(float).eps, 'gtol': 1e-8,
                },
            )

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(self, t):
        """t: array-like → x, xt  each (N, n_dof) numpy arrays."""
        t_tf = tf.constant(np.asarray(t, dtype=np.float32).reshape(-1, 1))
        x, x_t, _ = self._net_u(t_tf)
        return x.numpy(), x_t.numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Energy helper
# ═══════════════════════════════════════════════════════════════════════════════

def total_energy(x, xt, m=M_VAL, k=K_VAL):
    """x, xt: (n_time, n_dof) → E: (n_time,)"""
    Ek = 0.5 * m * np.sum(xt**2, axis=1)
    Ep = 0.5 * k  * np.sum((x[:, 1:] - x[:, :-1])**2, axis=1)
    return Ek + Ep


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('TensorFlow:', tf.__version__)
    os.makedirs(SAVE_DIR, exist_ok=True)

    M, C, K = build_matrices()

    for case, v_in in INPUT_VELOCITY_CASES.items():
        print(f'\n{"="*70}')
        print(f'Case: {case:>6s}  |  v0 = {v_in:>6.2f} m/s'
              f'  |  E0 = {0.5*M_VAL*v_in**2:.4f} J')
        print(f'{"="*70}')

        x0, xt0 = left_velocity_ic(N_DOF, v_in)

        # ── Newmark-beta ───────────────────────────────────────────────────────
        print('\n[Newmark-beta]')
        t0w = time.time()
        t_nm, x_nm, xt_nm = newmark_beta(M, C, K, x0, xt0)
        print(f'  Done in {time.time()-t0w:.2f} s   steps={len(t_nm)}')
        E_nm = total_energy(x_nm, xt_nm)

        # ── PINN (single network, full 10 s) ──────────────────────────────────
        print('\n[PINN — single network on full interval]')
        t0w = time.time()
        pinn = PINNLinearChain(T_END, x0, xt0, M, C, K)
        pinn.train(n_iter=N_ITER_ADAM)
        train_time = time.time() - t0w
        print(f'  Training done in {train_time:.2f} s')

        # dense query on same time grid as Newmark for fair comparison
        x_pinn, xt_pinn = pinn.predict(t_nm)
        E_pinn = total_energy(x_pinn, xt_pinn)

        # ── comparison plots ───────────────────────────────────────────────────

        # displacement per selected DOF
        fig, axes = plt.subplots(len(SELECTED_DOFS), 1,
                                 figsize=(12, 3*len(SELECTED_DOFS)),
                                 sharex=True)
        for ax, di in zip(axes, SELECTED_DOFS):
            ax.plot(t_nm, x_nm[:, di],
                    'k-', lw=1.5, alpha=0.8, label='Newmark-β')
            ax.plot(t_nm, x_pinn[:, di],
                    'C1--', lw=1.2, label='PINN')
            ax.set_ylabel(f'DOF {di+1}  x (m)')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(
            f'Displacement: PINN vs Newmark-β'
            f'  |  {case.capitalize()}  v0={v_in:.1f} m/s',
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'displacement_comparison_{case}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

        # velocity per selected DOF
        fig, axes = plt.subplots(len(SELECTED_DOFS), 1,
                                 figsize=(12, 3*len(SELECTED_DOFS)),
                                 sharex=True)
        for ax, di in zip(axes, SELECTED_DOFS):
            ax.plot(t_nm, xt_nm[:, di],
                    'k-', lw=1.5, alpha=0.8, label='Newmark-β')
            ax.plot(t_nm, xt_pinn[:, di],
                    'C1--', lw=1.2, label='PINN')
            ax.set_ylabel(f'DOF {di+1}  ẋ (m/s)')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(
            f'Velocity: PINN vs Newmark-β'
            f'  |  {case.capitalize()}  v0={v_in:.1f} m/s',
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'velocity_comparison_{case}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

        # energy conservation
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_nm, E_nm,   'k-',   lw=1.5, alpha=0.8, label='Newmark-β')
        ax.plot(t_nm, E_pinn, 'C1--', lw=1.2, label='PINN')
        ax.axhline(E_nm[0], color='grey', ls=':', lw=0.8, label=f'E0={E_nm[0]:.4f} J')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Total Energy (J)')
        ax.set_title(
            f'Energy conservation  |  {case.capitalize()}  v0={v_in:.1f} m/s'
        )
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'energy_comparison_{case}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

        # PINN loss history
        if pinn.loss_log:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.semilogy(pinn.loss_log, lw=1.2, color='C0')
            ax.set_xlabel('Logged iteration')
            ax.set_ylabel('Loss')
            ax.set_title(f'PINN training loss  |  {case.capitalize()}')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            path = os.path.join(SAVE_DIR, f'loss_history_{case}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {path}')

        # ── point-wise L2 error ────────────────────────────────────────────────
        err = np.sqrt(np.mean((x_pinn - x_nm)**2, axis=1))
        print(f'  Mean L2 error (displacement): {np.mean(err):.4e} m')
        print(f'  Max  L2 error (displacement): {np.max(err):.4e} m')

        # ── save data ──────────────────────────────────────────────────────────
        stem = f'linear_chain_{case}'
        np.savez(
            os.path.join(SAVE_DIR, stem + '.npz'),
            t=t_nm,
            x_nm=x_nm,   xt_nm=xt_nm,   E_nm=E_nm,
            x_pinn=x_pinn, xt_pinn=xt_pinn, E_pinn=E_pinn,
            v_in=np.array(v_in),
            train_time_s=np.array(train_time),
        )
        if _HAS_SAVEMAT:
            savemat(
                os.path.join(SAVE_DIR, stem + '.mat'),
                {
                    't': t_nm,
                    'x_nm': x_nm, 'xt_nm': xt_nm, 'E_nm': E_nm,
                    'x_pinn': x_pinn, 'xt_pinn': xt_pinn, 'E_pinn': E_pinn,
                    'v_in': np.array([[v_in]]),
                    'train_time_s': np.array([[train_time]]),
                }
            )

    print(f'\nAll results saved to  {SAVE_DIR}/')
    print('MATLAB export:', 'yes' if _HAS_SAVEMAT else 'no (scipy not found)')


if __name__ == '__main__':
    main()
