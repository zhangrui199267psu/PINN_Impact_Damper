"""
pinn_linear_chain.py
--------------------
PINN solver for a free-free linear chain with NO internal impact-damper masses.

System:  M x_tt + C x_t + K x = 0
  - 20 DOFs, free-free nearest-neighbour chain
  - Left-end initial velocity excitation
  - Time-marching segmentation for long-time accuracy
  - PINN results compared against Newmark-beta reference

This script is fully self-contained and requires only:
    TensorFlow >= 2.x, NumPy, SciPy, Matplotlib

All output plots and data are saved to  Results_Linear_Chain_PINN/.

Run:
    python pinn_linear_chain.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.optimize import minimize

# optional MATLAB export
try:
    from scipy.io import savemat
    _HAS_SAVEMAT = True
except ImportError:
    _HAS_SAVEMAT = False

# ── GPU / CPU selection ────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # set to '-1' to force CPU
np.random.seed(1234)
tf.random.set_seed(1234)

# ── physical parameters ────────────────────────────────────────────────────────
N_DOF  = 20
M_VAL  = 1.0    # mass per DOF (kg)
K_VAL  = 1.0    # spring stiffness (N/m)
C_VAL  = 0.0    # damping coefficient (undamped)

# ── PINN / training controls ───────────────────────────────────────────────────
LAYERS          = [1, 64, N_DOF]   # network architecture
WEIGHT_IC       = 1.0              # IC loss weight (beta_ic)
WEIGHT_ODE      = 1.0              # ODE residual loss weight (beta_ode)
T_SEGMENT       = 1.0              # segment duration (s)
N_COLLOC        = 100              # collocation points per segment
N_ITER_ADAM     = 1000             # Adam iterations per segment
USE_LBFGS       = True             # polish with L-BFGS-B after Adam
T_END           = 10.0             # total simulation time (s)

# ── input-energy cases ─────────────────────────────────────────────────────────
INPUT_VELOCITY_CASES = {
    'low':    -1.0,
    'medium': -2.0,
    'high':   -10.0,
}

SELECTED_DOFS = [0, 4, 9, 14, 19]   # 0-indexed, for plotting
SAVE_DIR      = 'Results_Linear_Chain_PINN'


# ═══════════════════════════════════════════════════════════════════════════════
# Keras-version-safe Adam factory
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
# System matrix builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_free_free_chain_matrices(n=N_DOF, m=M_VAL, k=K_VAL, c=C_VAL):
    """Build M, C, K for a uniform free-free nearest-neighbour chain."""
    M = m * np.eye(n)
    K = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            K[i, i]   += k;  K[i, i-1] -= k
            C[i, i]   += c;  C[i, i-1] -= c
        if i < n - 1:
            K[i, i]   += k;  K[i, i+1] -= k
            C[i, i]   += c;  C[i, i+1] -= c
    return M, C, K


def make_left_velocity_ic(n=N_DOF, v0=1.0):
    """Left-end velocity IC: x_i(0)=0, x_dot_1(0)=v0."""
    x0  = np.zeros((1, n))
    xt0 = np.zeros((1, n))
    xt0[0, 0] = float(v0)
    return x0, xt0


# ═══════════════════════════════════════════════════════════════════════════════
# Newmark-beta reference integrator
# ═══════════════════════════════════════════════════════════════════════════════

def newmark_beta(M, C, K, F, dt, n_steps, x0=None, xt0=None,
                 beta=0.25, gamma=0.5):
    n = M.shape[0]
    x   = np.zeros((n, n_steps))
    xt  = np.zeros((n, n_steps))
    xtt = np.zeros((n, n_steps))

    if x0  is not None: x[:,  0] = np.asarray(x0).flatten()[:n]
    if xt0 is not None: xt[:, 0] = np.asarray(xt0).flatten()[:n]

    xtt[:, 0] = np.linalg.solve(M, F[:, 0] - C @ xt[:, 0] - K @ x[:, 0])

    K_eff = M / (beta * dt**2) + gamma * C / (beta * dt) + K
    K_inv = np.linalg.inv(K_eff)

    for i in range(1, n_steps):
        F_eff = (
            F[:, i]
            + M @ (x[:, i-1] / (beta*dt**2) + xt[:, i-1] / (beta*dt)
                   + xtt[:, i-1] * (0.5/beta - 1.0))
            + C @ (gamma * x[:, i-1] / (beta*dt)
                   - xt[:, i-1] * (1.0 - gamma/beta)
                   - dt * xtt[:, i-1] * (1.0 - gamma/(2.0*beta)))
        )
        x[:, i]   = K_inv @ F_eff
        xtt[:, i] = ((x[:, i] - x[:, i-1]) / (beta*dt**2)
                     - xt[:, i-1] / (beta*dt)
                     - xtt[:, i-1] * (0.5/beta - 1.0))
        xt[:, i]  = xt[:, i-1] + dt * ((1.0 - gamma)*xtt[:, i-1]
                                        + gamma*xtt[:, i])
    return x, xt, xtt


# ═══════════════════════════════════════════════════════════════════════════════
# PINN — linear chain (no impact parameters)
# ═══════════════════════════════════════════════════════════════════════════════

class PINNLinearChain:
    """
    PINN for a free-free linear chain:  M x_tt + C x_t + K x = 0

    Network:  t  →  (x_1, ..., x_n)
    Loss:     beta_ic * L_ic  +  beta_ode * L_ode

    One instance is trained per time segment; ICs are propagated between
    segments to achieve long-time accuracy.

    Parameters
    ----------
    lb, ub       : (1,) array — time-domain bounds for input normalisation
    t0_data      : (1,1) — IC time (always 0 within a segment)
    t_colloc     : (N,1) — collocation times within the segment
    x0           : (1, n_dof) — displacement IC
    xt0          : (1, n_dof) — velocity IC
    M, C, K      : (n_dof, n_dof) system matrices
    layers       : list[int] — network architecture
    beta_ic      : float — IC loss weight
    beta_ode     : float — ODE residual loss weight
    use_lbfgs    : bool — run L-BFGS-B polishing after Adam
    """

    def __init__(self, lb, ub, t0_data, t_colloc,
                 x0, xt0, M, C, K, layers,
                 beta_ic=1.0, beta_ode=1.0, use_lbfgs=True):

        self.lb = tf.constant(np.asarray(lb, dtype=np.float32).reshape(1, -1))
        self.ub = tf.constant(np.asarray(ub, dtype=np.float32).reshape(1, -1))

        self.t_colloc = tf.constant(
            np.asarray(t_colloc, dtype=np.float32).reshape(-1, 1)
        )
        self.t0_data  = tf.constant(
            np.asarray(t0_data,  dtype=np.float32).reshape(-1, 1)
        )
        self.x0_data  = tf.constant(np.asarray(x0,  dtype=np.float32))
        self.xt0_data = tf.constant(np.asarray(xt0, dtype=np.float32))

        self.M_tf = tf.constant(M, dtype=tf.float32)
        self.C_tf = tf.constant(C, dtype=tf.float32)
        self.K_tf = tf.constant(K, dtype=tf.float32)

        self.beta_ic  = float(beta_ic)
        self.beta_ode = float(beta_ode)
        self.n_dof    = M.shape[0]

        self.layers = layers
        self.weights, self.biases = self._init_nn(layers)
        self.trainable_vars = self.weights + self.biases

        self.adam       = _make_adam(lr=1e-3)
        self.use_lbfgs  = bool(use_lbfgs)
        self.loss_log   = []

    # ── network construction ───────────────────────────────────────────────────
    def _init_nn(self, layers):
        weights, biases = [], []
        for l in range(len(layers) - 1):
            stddev = np.sqrt(2.0 / (layers[l] + layers[l+1]))
            W = tf.Variable(
                tf.random.truncated_normal(
                    [layers[l], layers[l+1]], stddev=stddev, dtype=tf.float32
                )
            )
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def _neural_net(self, t):
        H = 2.0 * (t - self.lb) / (self.ub - self.lb) - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]

    # ── physics via nested GradientTape + batch_jacobian ──────────────────────
    def _net_u(self, t):
        """Return x, x_t, x_tt — each (N, n_dof)."""
        with tf.GradientTape() as tape2:
            tape2.watch(t)
            with tf.GradientTape() as tape1:
                tape1.watch(t)
                x    = self._neural_net(t)
            x_t  = tf.squeeze(tape1.batch_jacobian(x,   t), axis=-1)
        x_tt = tf.squeeze(tape2.batch_jacobian(x_t, t), axis=-1)
        return x, x_t, x_tt

    def _net_ode_residual(self, t):
        """ODE residual: M x_tt + C x_t + K x  (should be zero)."""
        x, x_t, x_tt = self._net_u(t)
        residual = (
            tf.transpose(tf.matmul(self.M_tf, x_tt, transpose_b=True))
            + tf.transpose(tf.matmul(self.C_tf, x_t,  transpose_b=True))
            + tf.transpose(tf.matmul(self.K_tf, x,    transpose_b=True))
        )
        return residual

    # ── loss ──────────────────────────────────────────────────────────────────
    def _compute_loss(self):
        x0_pred, xt0_pred, _ = self._net_u(self.t0_data)
        loss_ic  = (tf.reduce_mean(tf.square(x0_pred  - self.x0_data))
                    + tf.reduce_mean(tf.square(xt0_pred - self.xt0_data)))
        loss_ode = tf.reduce_mean(tf.square(self._net_ode_residual(self.t_colloc)))
        return self.beta_ic * loss_ic + self.beta_ode * loss_ode, loss_ic, loss_ode

    # ── Adam step ─────────────────────────────────────────────────────────────
    @tf.function
    def _adam_step(self):
        with tf.GradientTape() as tape:
            loss, l_ic, l_ode = self._compute_loss()
        grads = tape.gradient(loss, self.trainable_vars)
        self.adam.apply_gradients(zip(grads, self.trainable_vars))
        return loss, l_ic, l_ode

    # ── L-BFGS-B helpers ──────────────────────────────────────────────────────
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
            loss, l_ic, l_ode = self._compute_loss()
        grads = tape.gradient(loss, self.trainable_vars)
        return loss, l_ic, l_ode, grads

    def _lbfgs_objective(self, flat_f64):
        self._set_flat_params(flat_f64.astype(np.float32))
        loss, l_ic, l_ode, grads = self._loss_and_grads_tf()
        lv = float(loss)
        self.loss_log.append(lv)
        print(f'  L-BFGS  Loss {lv:.4e}  IC {float(l_ic):.4e}'
              f'  ODE {float(l_ode):.4e}')
        grad_flat = np.concatenate(
            [g.numpy().flatten() for g in grads]
        ).astype(np.float64)
        return lv, grad_flat

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, n_iter=N_ITER_ADAM, print_every=200):
        t_wall = time.time()
        for it in range(n_iter):
            loss, l_ic, l_ode = self._adam_step()
            if it % print_every == 0:
                lv = float(loss)
                self.loss_log.append(lv)
                print(f'  Adam it {it:5d}  Loss {lv:.4e}'
                      f'  IC {float(l_ic):.4e}  ODE {float(l_ode):.4e}'
                      f'  {time.time()-t_wall:.1f}s')
                t_wall = time.time()

        if self.use_lbfgs:
            x0_flat = self._get_flat_params().astype(np.float64)
            minimize(
                self._lbfgs_objective, x0_flat,
                method='L-BFGS-B', jac=True,
                options={
                    'maxiter': 50000, 'maxfun': 50000,
                    'maxcor': 50, 'maxls': 50,
                    'ftol': np.finfo(float).eps, 'gtol': 1e-8,
                },
            )

    # ── prediction ────────────────────────────────────────────────────────────
    def predict(self, t):
        """
        Query the network at times t (array-like).
        Returns x, xt, xtt — each (N, n_dof) numpy arrays.
        """
        t_tf = tf.constant(np.asarray(t, dtype=np.float32).reshape(-1, 1))
        x, x_t, x_tt = self._net_u(t_tf)
        return x.numpy(), x_t.numpy(), x_tt.numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Time-marching PINN driver  (no impact detection — purely linear chain)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pinn_linear_chain(
    M, C, K,
    x0_init, xt0_init,
    t_end=T_END,
    t_segment=T_SEGMENT,
    n_colloc=N_COLLOC,
    n_iter=N_ITER_ADAM,
    use_lbfgs=USE_LBFGS,
    case_name='case',
):
    """
    Simulate the linear chain using a time-marching PINN.

    The interval [0, t_end] is split into fixed segments of duration
    t_segment.  Each segment trains an independent PINN network whose
    ICs are the predicted state at the end of the previous segment.

    Returns a dict with keys: t, x, xt, segment_train_times_s, total_train_time_s
    """
    lb = np.array([0.0])
    ub = np.array([float(t_segment)])
    t_colloc = np.linspace(0.0, float(t_segment), int(n_colloc)).reshape(-1, 1)

    x0_cur  = x0_init.copy()   # (1, n_dof)
    xt0_cur = xt0_init.copy()  # (1, n_dof)

    t_global     = 0.0
    t_hist, x_hist, xt_hist = [], [], []
    seg_times    = []
    t_total_start = time.perf_counter()
    seg_id        = 0

    while t_global < float(t_end) - 1e-10:
        seg_id += 1

        # clip last segment to land exactly on t_end
        t_remaining = float(t_end) - t_global
        t_dur       = min(float(t_segment), t_remaining)

        # adjust ub and collocation if last segment is shorter
        ub_cur      = np.array([t_dur])
        t_col_cur   = np.linspace(0.0, t_dur, int(n_colloc)).reshape(-1, 1)

        print(f'\n[{case_name}] Segment {seg_id:03d}  '
              f't_global = {t_global:.3f} → {t_global + t_dur:.3f} s')

        seg_start = time.perf_counter()
        model = PINNLinearChain(
            lb, ub_cur,
            np.array([[0.0]]), t_col_cur,
            x0_cur, xt0_cur,
            M, C, K,
            LAYERS,
            beta_ic=WEIGHT_IC, beta_ode=WEIGHT_ODE,
            use_lbfgs=use_lbfgs,
        )
        model.train(n_iter=int(n_iter))
        seg_time = time.perf_counter() - seg_start
        seg_times.append(seg_time)

        # dense output over this segment
        t_dense = np.linspace(0.0, t_dur, int(n_colloc) + 1).reshape(-1, 1)
        x_seg, xt_seg, _ = model.predict(t_dense)

        # drop duplicate boundary point when stitching
        t_global_dense = t_global + t_dense.flatten()
        if len(t_hist) > 0:
            t_global_dense = t_global_dense[1:]
            x_seg  = x_seg[1:]
            xt_seg = xt_seg[1:]

        t_hist.append(t_global_dense)
        x_hist.append(x_seg)
        xt_hist.append(xt_seg)

        # propagate ICs from the end of this segment
        x_end, xt_end, _ = model.predict(np.array([[t_dur]]))
        x0_cur  = x_end.copy()    # (1, n_dof)
        xt0_cur = xt_end.copy()   # (1, n_dof)

        t_global += t_dur
        print(f'  Segment done in {seg_time:.2f} s')

    total_train_time = time.perf_counter() - t_total_start

    return {
        't':                   np.concatenate(t_hist),
        'x':                   np.vstack(x_hist),
        'xt':                  np.vstack(xt_hist),
        'n_segments':          seg_id,
        'segment_train_times_s': np.array(seg_times),
        'total_train_time_s':  total_train_time,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Energy helper
# ═══════════════════════════════════════════════════════════════════════════════

def total_energy_from_trajectories(x, xt, m=M_VAL, k=K_VAL):
    """x, xt: (n_time, n_dof) — returns (n_time,)."""
    Ek         = 0.5 * m * np.sum(xt**2, axis=1)
    spring_rel = x[:, 1:] - x[:, :-1]
    Ep         = 0.5 * k  * np.sum(spring_rel**2, axis=1)
    return Ek + Ep


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('TensorFlow version:', tf.__version__)
    os.makedirs(SAVE_DIR, exist_ok=True)

    M, C, K = build_free_free_chain_matrices()

    # Newmark reference (for comparison)
    dt_ref   = 0.001
    n_ref    = int(round(T_END / dt_ref)) + 1
    t_ref    = np.linspace(0.0, T_END, n_ref)
    F_ref    = np.zeros((N_DOF, n_ref))

    pinn_results = {}

    for case, v_in in INPUT_VELOCITY_CASES.items():
        print(f'\n{"="*70}')
        print(f'Case: {case:>6s}  |  v0 = {v_in:>6.2f} m/s'
              f'  |  E0 = {0.5*M_VAL*v_in**2:.4f} J')
        print(f'{"="*70}')

        x0_init, xt0_init = make_left_velocity_ic(N_DOF, v_in)

        # ── PINN simulation ────────────────────────────────────────────────────
        res = run_pinn_linear_chain(
            M, C, K,
            x0_init, xt0_init,
            t_end=T_END,
            case_name=case,
        )
        res['v_in'] = v_in
        res['E']    = total_energy_from_trajectories(res['x'], res['xt'])
        pinn_results[case] = res

        print(f'\n[{case}] Done: {res["n_segments"]} segments, '
              f'total train time = {res["total_train_time_s"]:.2f} s')

        # ── Newmark reference for this case ────────────────────────────────────
        x0_flat  = x0_init.flatten()
        xt0_flat = xt0_init.flatten()
        x_ref, xt_ref, _ = newmark_beta(
            M, C, K, F_ref, dt_ref, n_ref,
            x0=x0_flat, xt0=xt0_flat,
        )

        # ── comparison plot ────────────────────────────────────────────────────
        fig, axes = plt.subplots(len(SELECTED_DOFS), 1,
                                 figsize=(12, 3 * len(SELECTED_DOFS)),
                                 sharex=True)
        for ax, dof_idx in zip(axes, SELECTED_DOFS):
            ax.plot(t_ref, x_ref[dof_idx, :],
                    'k-', lw=1.5, alpha=0.7, label='Newmark-β')
            ax.plot(res['t'], res['x'][:, dof_idx],
                    'C1--', lw=1.2, label='PINN')
            ax.set_ylabel(f'DOF {dof_idx+1}  x (m)')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(
            f'PINN vs Newmark-β  |  {case.capitalize()}'
            f'  $v_0$={v_in:.1f} m/s  —  Free-free linear chain',
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'comparison_{case}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {path}')

    # ── summary: displacement for all cases ───────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (case, res) in zip(axes, pinn_results.items()):
        for i in SELECTED_DOFS:
            ax.plot(res['t'], res['x'][:, i], lw=1.2, label=f'DOF {i+1}')
        ax.set_title(f"{case.capitalize()}  —  $v_0$={res['v_in']:.1f} m/s")
        ax.set_ylabel('Displacement (m)')
        ax.legend(ncol=5, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('PINN  |  Free-free linear chain (no impact damper)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'pinn_displacement_all_cases.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {path}')

    # ── energy conservation ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (case, res) in zip(axes, pinn_results.items()):
        ax.plot(res['t'], res['E'], lw=1.5, color='C2')
        E0 = res['E'][0]
        ax.axhline(E0, color='k', ls='--', lw=0.8, label=f'E0 = {E0:.4f} J')
        ax.set_title(f"{case.capitalize()}  —  $v_0$={res['v_in']:.1f} m/s")
        ax.set_ylabel('Total Energy (J)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('PINN energy conservation — Free-free linear chain',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'pinn_energy_all_cases.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')

    # ── training time summary ──────────────────────────────────────────────────
    print('\n--- Training summary ---')
    for case, res in pinn_results.items():
        print(f"  {case:>6s}: segments={res['n_segments']:3d}, "
              f"total_train={res['total_train_time_s']:.2f} s, "
              f"mean_seg={np.mean(res['segment_train_times_s']):.2f} s")

    # ── save numerical data ────────────────────────────────────────────────────
    for case, res in pinn_results.items():
        stem = f'pinn_linear_chain_{case}'
        np.savez(
            os.path.join(SAVE_DIR, stem + '.npz'),
            t=res['t'], x=res['x'], xt=res['xt'],
            E=res['E'],
            v_in=np.array(res['v_in']),
            total_train_time_s=np.array(res['total_train_time_s']),
            segment_train_times_s=res['segment_train_times_s'],
        )
        if _HAS_SAVEMAT:
            savemat(
                os.path.join(SAVE_DIR, stem + '.mat'),
                {
                    't':   res['t'],
                    'x':   res['x'],
                    'xt':  res['xt'],
                    'E':   res['E'],
                    'v_in': np.array([[res['v_in']]]),
                    'total_train_time_s': np.array([[res['total_train_time_s']]]),
                }
            )

    print(f'\nAll data saved to  {SAVE_DIR}/')
    print('MATLAB export:', 'yes' if _HAS_SAVEMAT else 'no (scipy not found)')


if __name__ == '__main__':
    main()
