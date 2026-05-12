"""
newmark_linear_chain.py
-----------------------
Newmark-beta reference solver for a free-free linear chain with NO internal
impact-damper masses.

System:  M x_tt + C x_t + K x = 0
  - 20 DOFs, free-free nearest-neighbour chain
  - Left-end initial velocity excitation  (x_i(0) = 0, x_dot_1(0) = v0)
  - Three input-energy cases: low / medium / high

All plots are saved to  Results_Linear_Chain/  so the script runs cleanly in
a headless terminal.

Run:
    python newmark_linear_chain.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# optional MATLAB export
try:
    from scipy.io import savemat
    _HAS_SAVEMAT = True
except ImportError:
    _HAS_SAVEMAT = False

# ── reproducibility ────────────────────────────────────────────────────────────
np.random.seed(1234)

# ── physical parameters ────────────────────────────────────────────────────────
N_DOF  = 20
M_VAL  = 1.0   # mass per DOF (kg)
K_VAL  = 1.0   # spring stiffness (N/m)
C_VAL  = 0.0   # damping coefficient (undamped)

# ── simulation controls ────────────────────────────────────────────────────────
T_END  = 10.0   # total simulation time (s)
DT     = 0.001  # time step (s)
BETA   = 0.25   # Newmark parameter (0.25 = average-acceleration, unconditionally stable)
GAMMA  = 0.5    # Newmark parameter

# ── post-processing ────────────────────────────────────────────────────────────
SELECTED_DOFS = [0, 4, 9, 14, 19]   # 0-indexed
INPUT_VELOCITY_CASES = {
    'low':    -1.0,
    'medium': -2.0,
    'high':   -10.0,
}
SAVE_DIR = 'Results_Linear_Chain'


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def build_free_free_chain_matrices(n=N_DOF, m=M_VAL, k=K_VAL, c=C_VAL):
    """
    Build M, C, K for a uniform free-free nearest-neighbour chain.

    K topology (free-free):
      diag  = [k, 2k, ..., 2k, k]
      off-diag = -k for nearest neighbours
    """
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
    """Left-end velocity excitation IC: x_i(0)=0, x_dot_1(0)=v0."""
    x0  = np.zeros(n)
    xt0 = np.zeros(n)
    xt0[0] = float(v0)
    return x0, xt0


def newmark_beta(M, C, K, F, dt, n_steps, x0=None, xt0=None,
                 beta=BETA, gamma=GAMMA):
    """
    Implicit Newmark-beta integrator.

    Parameters
    ----------
    M, C, K   : (n, n) system matrices
    F         : (n, n_steps) external force matrix (zero for free vibration)
    dt        : time step
    n_steps   : number of time steps
    x0, xt0   : (n,) initial displacement and velocity

    Returns
    -------
    x, xt, xtt : (n, n_steps) displacement, velocity, acceleration histories
    """
    n = M.shape[0]
    x   = np.zeros((n, n_steps))
    xt  = np.zeros((n, n_steps))
    xtt = np.zeros((n, n_steps))

    if x0  is not None: x[:,  0] = np.asarray(x0).flatten()[:n]
    if xt0 is not None: xt[:, 0] = np.asarray(xt0).flatten()[:n]

    xtt[:, 0] = np.linalg.solve(
        M, F[:, 0] - C @ xt[:, 0] - K @ x[:, 0]
    )

    K_eff = M / (beta * dt**2) + gamma * C / (beta * dt) + K
    K_inv = np.linalg.inv(K_eff)

    for i in range(1, n_steps):
        F_eff = (
            F[:, i]
            + M @ (
                x[:, i-1] / (beta * dt**2)
                + xt[:, i-1] / (beta * dt)
                + xtt[:, i-1] * (0.5 / beta - 1.0)
            )
            + C @ (
                gamma * x[:, i-1] / (beta * dt)
                - xt[:, i-1] * (1.0 - gamma / beta)
                - dt * xtt[:, i-1] * (1.0 - gamma / (2.0 * beta))
            )
        )
        x[:, i]   = K_inv @ F_eff
        xtt[:, i] = (
            (x[:, i] - x[:, i-1]) / (beta * dt**2)
            - xt[:, i-1] / (beta * dt)
            - xtt[:, i-1] * (0.5 / beta - 1.0)
        )
        xt[:, i] = (
            xt[:, i-1]
            + dt * ((1.0 - gamma) * xtt[:, i-1] + gamma * xtt[:, i])
        )

    return x, xt, xtt


def total_energy(x, xt, m=M_VAL, k=K_VAL):
    """
    Compute total mechanical energy at every time step.

    x, xt : (n_dof, n_steps)
    Returns E : (n_steps,)
    """
    Ek          = 0.5 * m * np.sum(xt**2, axis=0)
    spring_rel  = x[1:, :] - x[:-1, :]
    Ep          = 0.5 * k  * np.sum(spring_rel**2, axis=0)
    return Ek + Ep


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    M, C, K = build_free_free_chain_matrices()
    n_steps = int(round(T_END / DT)) + 1
    t_vec   = np.linspace(0.0, T_END, n_steps)
    F       = np.zeros((N_DOF, n_steps))

    results = {}

    # ── run Newmark for each input-velocity case ───────────────────────────────
    for case, v_in in INPUT_VELOCITY_CASES.items():
        print(f'\n{"="*60}')
        print(f'Case: {case:>6s}  |  v0 = {v_in:>6.2f} m/s')
        print(f'{"="*60}')

        x0, xt0 = make_left_velocity_ic(N_DOF, v_in)
        x, xt, xtt = newmark_beta(M, C, K, F, DT, n_steps, x0=x0, xt0=xt0)
        E = total_energy(x, xt)

        drift = abs(E[-1] - E[0]) / max(abs(E[0]), 1e-12)
        print(f'  E0 = {E[0]:.6f} J')
        print(f'  E_final = {E[-1]:.6f} J')
        print(f'  Relative energy drift = {drift:.2e}')

        results[case] = {
            't': t_vec, 'x': x, 'xt': xt, 'xtt': xtt,
            'E': E, 'v_in': v_in,
        }

    # ── plot displacements ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (case, res) in zip(axes, results.items()):
        for i in SELECTED_DOFS:
            ax.plot(res['t'], res['x'][i, :], lw=1.2, label=f'DOF {i+1}')
        ax.set_title(f"{case.capitalize()}  —  $v_0$ = {res['v_in']:.1f} m/s")
        ax.set_ylabel('Displacement (m)')
        ax.legend(ncol=5, fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Newmark-β  |  Free-free linear chain (no impact damper)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'displacement_all_cases.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {path}')

    # ── plot velocities ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (case, res) in zip(axes, results.items()):
        for i in SELECTED_DOFS:
            ax.plot(res['t'], res['xt'][i, :], lw=1.2, label=f'DOF {i+1}')
        ax.set_title(f"{case.capitalize()}  —  $v_0$ = {res['v_in']:.1f} m/s")
        ax.set_ylabel('Velocity (m/s)')
        ax.legend(ncol=5, fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Newmark-β  |  Velocities — Free-free linear chain',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'velocity_all_cases.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')

    # ── plot energy conservation ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (case, res) in zip(axes, results.items()):
        ax.plot(res['t'], res['E'], lw=1.5, color='C2')
        E0 = res['E'][0]
        ax.axhline(E0, color='k', ls='--', lw=0.8, label=f'E0 = {E0:.4f} J')
        ax.set_title(f"{case.capitalize()}  —  $v_0$ = {res['v_in']:.1f} m/s")
        ax.set_ylabel('Total Energy (J)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Newmark-β  |  Energy conservation — Free-free linear chain',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'energy_all_cases.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')

    # ── wave propagation snapshot (space-time) ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (case, res) in zip(axes, results.items()):
        pcm = ax.pcolormesh(
            res['t'], np.arange(1, N_DOF + 1), res['x'],
            shading='auto', cmap='RdBu_r',
            vmin=-np.max(np.abs(res['x'])), vmax=np.max(np.abs(res['x']))
        )
        ax.set_title(f"{case.capitalize()}  $v_0$={res['v_in']:.1f} m/s")
        ax.set_xlabel('Time (s)')
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label='x (m)')
    axes[0].set_ylabel('DOF index')
    fig.suptitle('Space-time displacement map — Free-free linear chain',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'spacetime_all_cases.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')

    # ── save numerical data ────────────────────────────────────────────────────
    for case, res in results.items():
        stem = f'newmark_linear_chain_{case}'

        np.savez(
            os.path.join(SAVE_DIR, stem + '.npz'),
            t=res['t'], x=res['x'], xt=res['xt'], xtt=res['xtt'],
            E=res['E'], v_in=np.array(res['v_in']),
        )

        if _HAS_SAVEMAT:
            savemat(
                os.path.join(SAVE_DIR, stem + '.mat'),
                {
                    't':   res['t'],
                    'x':   res['x'],
                    'xt':  res['xt'],
                    'xtt': res['xtt'],
                    'E':   res['E'],
                    'v_in': np.array([[res['v_in']]]),
                }
            )

    print(f'\nAll data saved to  {SAVE_DIR}/')
    print('MATLAB export:', 'yes' if _HAS_SAVEMAT else 'no (scipy not found)')


if __name__ == '__main__':
    main()
