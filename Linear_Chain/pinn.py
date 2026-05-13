"""
pinn.py
-------
PINN solver for a free-free linear chain (no impact-damper masses).

A single neural network is trained on the full [0, T_END] interval with no
time-marching segmentation.

System:  M x_tt + C x_t + K x = 0
  - N_DOF = 20, free-free nearest-neighbour chain
  - Left-end initial velocity excitation: x_dot_1(0) = v0, all else zero
  - Three input-energy cases: low / medium / high

Outputs saved to  Results_Linear_Chain/PINN/:
  - pinn_{case}.npz   (t, x, xt, E, v_in, train_time_s)
  - pinn_{case}.mat   (if scipy available)
  - pinn_displacement_{case}.png
  - pinn_loss_{case}.png

Run:
    python pinn.py                  # default: 20 DOFs
    python pinn.py --ndof 10
    python pinn.py --ndof 40
"""

import os
import time
import argparse
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # set to '-1' to force CPU
np.random.seed(1234)
tf.random.set_seed(1234)

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════════

N_DOF  = 20
M_VAL  = 1.0
K_VAL  = 1.0
C_VAL  = 0.0

T_END  = 10.0

LAYERS      = [1, 128, 128, N_DOF]
N_COLLOC    = 2000
N_ITER_ADAM = 5000
USE_LBFGS   = True
WEIGHT_IC   = 10.0
WEIGHT_ODE  = 1.0

# output time grid (matches newmark.py DT=0.001 for easy comparison)
DT_OUT = 0.001

INPUT_VELOCITY_CASES = {
    'low':    -1.0,
    'medium': -2.0,
    'high':   -10.0,
}
SELECTED_DOFS = [0, 4, 9, 14, 19]
SAVE_DIR = os.path.join('Results_Linear_Chain', 'PINN')


# ═══════════════════════════════════════════════════════════════════════════════
# System
# ═══════════════════════════════════════════════════════════════════════════════

def build_matrices(n=N_DOF, m=M_VAL, k=K_VAL, c=C_VAL):
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
    x0  = np.zeros(n)
    xt0 = np.zeros(n)
    xt0[0] = float(v0)
    return x0, xt0


def total_energy(x, xt, m=M_VAL, k=K_VAL):
    Ek = 0.5 * m * np.sum(xt**2, axis=1)
    Ep = 0.5 * k  * np.sum((x[:, 1:] - x[:, :-1])**2, axis=1)
    return Ek + Ep


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
# PINN — single network on [0, T_END]
# ═══════════════════════════════════════════════════════════════════════════════

class PINNLinearChain:
    """
    Single neural network trained on the entire time interval [0, T_END].

    Network:  t  →  (x_1, ..., x_n)
    Loss:     w_ic * L_ic  +  w_ode * L_ode

    No time-marching; one training pass covers the full simulation horizon.
    """

    def __init__(self, t_end, x0, xt0, M, C, K,
                 layers=LAYERS, n_colloc=N_COLLOC,
                 w_ic=WEIGHT_IC, w_ode=WEIGHT_ODE,
                 use_lbfgs=USE_LBFGS):

        self.t_end     = float(t_end)
        self.n_dof     = M.shape[0]
        self.w_ic      = float(w_ic)
        self.w_ode     = float(w_ode)
        self.use_lbfgs = bool(use_lbfgs)

        self.lb = tf.constant([[0.0]],          dtype=tf.float32)
        self.ub = tf.constant([[self.t_end]],   dtype=tf.float32)

        t_col = np.linspace(0.0, self.t_end, n_colloc).reshape(-1, 1).astype(np.float32)
        self.t_col    = tf.constant(t_col)
        self.t0_data  = tf.constant([[0.0]], dtype=tf.float32)
        self.x0_data  = tf.constant(x0.reshape(1, -1),  dtype=tf.float32)
        self.xt0_data = tf.constant(xt0.reshape(1, -1), dtype=tf.float32)

        self.M_tf = tf.constant(M, dtype=tf.float32)
        self.C_tf = tf.constant(C, dtype=tf.float32)
        self.K_tf = tf.constant(K, dtype=tf.float32)

        self.weights, self.biases = self._init_nn(layers)
        self.trainable_vars = self.weights + self.biases
        self.adam     = _make_adam(lr=1e-3)
        self.loss_log = []      # (iteration_label, total, ic, ode)

    def _init_nn(self, layers):
        ws, bs = [], []
        for l in range(len(layers) - 1):
            stddev = np.sqrt(2.0 / (layers[l] + layers[l+1]))
            W = tf.Variable(tf.random.truncated_normal(
                [layers[l], layers[l+1]], stddev=stddev, dtype=tf.float32))
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32))
            ws.append(W); bs.append(b)
        return ws, bs

    def _net(self, t):
        H = 2.0 * (t - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]

    def _net_u(self, t):
        with tf.GradientTape() as t2:
            t2.watch(t)
            with tf.GradientTape() as t1:
                t1.watch(t)
                x   = self._net(t)
            x_t  = tf.squeeze(t1.batch_jacobian(x,   t), axis=-1)
        x_tt = tf.squeeze(t2.batch_jacobian(x_t, t), axis=-1)
        return x, x_t, x_tt

    def _compute_loss(self):
        x, x_t, x_tt = self._net_u(self.t_col)
        res = (
            tf.transpose(tf.matmul(self.M_tf, x_tt, transpose_b=True))
            + tf.transpose(tf.matmul(self.C_tf, x_t,  transpose_b=True))
            + tf.transpose(tf.matmul(self.K_tf, x,    transpose_b=True))
        )
        loss_ode = tf.reduce_mean(tf.square(res))

        x0_p, xt0_p, _ = self._net_u(self.t0_data)
        loss_ic = (tf.reduce_mean(tf.square(x0_p  - self.x0_data))
                   + tf.reduce_mean(tf.square(xt0_p - self.xt0_data)))

        return self.w_ic * loss_ic + self.w_ode * loss_ode, loss_ic, loss_ode

    @tf.function
    def _adam_step(self):
        with tf.GradientTape() as tape:
            loss, l_ic, l_ode = self._compute_loss()
        grads = tape.gradient(loss, self.trainable_vars)
        self.adam.apply_gradients(zip(grads, self.trainable_vars))
        return loss, l_ic, l_ode

    def _get_flat(self):
        return np.concatenate([v.numpy().flatten() for v in self.trainable_vars])

    def _set_flat(self, f32):
        idx = 0
        for v in self.trainable_vars:
            n = int(np.prod(v.shape))
            v.assign(tf.reshape(f32[idx:idx+n], v.shape))
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
        self.loss_log.append(('lbfgs', lv, float(l_ic), float(l_ode)))
        print(f'  L-BFGS  total {lv:.4e}  IC {float(l_ic):.4e}'
              f'  ODE {float(l_ode):.4e}')
        return lv, np.concatenate(
            [g.numpy().flatten() for g in grads]).astype(np.float64)

    def train(self, n_iter=N_ITER_ADAM, print_every=500):
        t_wall = time.time()
        for it in range(n_iter):
            loss, l_ic, l_ode = self._adam_step()
            if it % print_every == 0:
                lv = float(loss)
                self.loss_log.append((it, lv, float(l_ic), float(l_ode)))
                print(f'  Adam {it:6d}  total {lv:.4e}'
                      f'  IC {float(l_ic):.4e}  ODE {float(l_ode):.4e}'
                      f'  ({time.time()-t_wall:.1f}s)')
                t_wall = time.time()

        if self.use_lbfgs:
            print('\n  Starting L-BFGS-B polishing...')
            minimize(self._lbfgs_obj, self._get_flat().astype(np.float64),
                     method='L-BFGS-B', jac=True,
                     options={'maxiter': 50000, 'maxfun': 50000,
                              'maxcor': 50, 'maxls': 50,
                              'ftol': np.finfo(float).eps, 'gtol': 1e-8})

    def predict(self, t):
        """t : array-like  →  x, xt  each (N, n_dof) numpy arrays."""
        t_tf = tf.constant(np.asarray(t, dtype=np.float32).reshape(-1, 1))
        x, x_t, _ = self._net_u(t_tf)
        return x.numpy(), x_t.numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='PINN solver for a free-free linear chain.'
    )
    parser.add_argument('--ndof', type=int, default=N_DOF,
                        help='Number of DOFs (default: %(default)s)')
    args = parser.parse_args()

    n_dof = args.ndof
    if n_dof < 2:
        parser.error('--ndof must be >= 2')

    # 5 evenly-spaced DOFs to plot, always including the first and last
    n_plot = min(5, n_dof)
    selected_dofs = list(dict.fromkeys(
        np.linspace(0, n_dof - 1, n_plot, dtype=int).tolist()
    ))

    # output layer size must match n_dof
    layers = [1, 128, 128, n_dof]

    print('TensorFlow:', tf.__version__)
    print(f'N_DOF = {n_dof}  |  layers = {layers}')
    print(f'Plotting DOFs: {[d+1 for d in selected_dofs]}')
    os.makedirs(SAVE_DIR, exist_ok=True)

    M, C, K = build_matrices(n=n_dof)
    t_out   = np.linspace(0.0, T_END, int(round(T_END / DT_OUT)) + 1)

    for case, v_in in INPUT_VELOCITY_CASES.items():
        print(f'\n{"="*60}')
        print(f'  Case: {case}  |  v0 = {v_in:.2f} m/s'
              f'  |  E0 = {0.5*M_VAL*v_in**2:.4f} J')
        print(f'{"="*60}')

        x0, xt0 = left_velocity_ic(n=n_dof, v0=v_in)

        t_start = time.time()
        pinn = PINNLinearChain(T_END, x0, xt0, M, C, K, layers=layers)
        pinn.train(n_iter=N_ITER_ADAM)
        train_time = time.time() - t_start
        print(f'\n  Training done in {train_time:.2f} s')

        x, xt = pinn.predict(t_out)
        E = total_energy(x, xt)

        drift = abs(E[-1] - E[0]) / max(abs(E[0]), 1e-12)
        print(f'  E0={E[0]:.6f} J   E_final={E[-1]:.6f} J'
              f'   relative drift={drift:.2e}')

        tag = f'{n_dof}dof_{case}'

        # ── displacement plot ──────────────────────────────────────────────────
        fig, axes = plt.subplots(len(selected_dofs), 1,
                                 figsize=(12, 3*len(selected_dofs)),
                                 sharex=True)
        for ax, di in zip(axes, selected_dofs):
            ax.plot(t_out, x[:, di], 'C1-', lw=1.2)
            ax.set_ylabel(f'DOF {di+1}  x (m)')
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'PINN displacement  |  {n_dof} DOFs  |  '
                     f'{case.capitalize()}  v0={v_in:.1f} m/s', fontsize=12)
        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'pinn_displacement_{tag}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

        # ── loss history ───────────────────────────────────────────────────────
        if pinn.loss_log:
            log = np.array([row[1] for row in pinn.loss_log])
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.semilogy(log, lw=1.2, color='C0')
            ax.set_xlabel('Logged step')
            ax.set_ylabel('Loss')
            ax.set_title(f'PINN training loss  |  {n_dof} DOFs  |  '
                         f'{case.capitalize()}')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            path = os.path.join(SAVE_DIR, f'pinn_loss_{tag}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {path}')

        # ── save data ──────────────────────────────────────────────────────────
        stem = f'pinn_{tag}'
        np.savez(os.path.join(SAVE_DIR, stem + '.npz'),
                 t=t_out, x=x, xt=xt, E=E,
                 v_in=np.array(v_in),
                 n_dof=np.array(n_dof),
                 train_time_s=np.array(train_time))
        if _HAS_SAVEMAT:
            savemat(os.path.join(SAVE_DIR, stem + '.mat'),
                    {'t': t_out, 'x': x, 'xt': xt, 'E': E,
                     'v_in': np.array([[v_in]]),
                     'n_dof': np.array([[n_dof]]),
                     'train_time_s': np.array([[train_time]])})
        print(f'  Data saved: {stem}.npz'
              + (' + .mat' if _HAS_SAVEMAT else ''))

    print(f'\nDone. Results in {SAVE_DIR}/')


if __name__ == '__main__':
    main()
