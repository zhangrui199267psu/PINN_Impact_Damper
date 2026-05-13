"""
newmark.py
----------
Newmark-beta solver for a free-free linear chain (no impact-damper masses).

System:  M x_tt + C x_t + K x = 0
  - N_DOF = 20, free-free nearest-neighbour chain
  - Left-end initial velocity excitation: x_dot_1(0) = v0, all else zero
  - Three input-energy cases: low / medium / high

Outputs saved to  Results_Linear_Chain/Newmark/:
  - newmark_{case}.npz   (t, x, xt, E, v_in)
  - newmark_{case}.mat   (if scipy available)
  - displacement_{case}.png
  - velocity_{case}.png
  - energy_{case}.png

Run:
    python newmark.py                  # default: 20 DOFs
    python newmark.py --ndof 10
    python newmark.py --ndof 40
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy.io import savemat
    _HAS_SAVEMAT = True
except ImportError:
    _HAS_SAVEMAT = False

np.random.seed(1234)

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════════

N_DOF  = 20
M_VAL  = 1.0    # mass per DOF (kg)
K_VAL  = 1.0    # spring stiffness (N/m)
C_VAL  = 0.0    # damping coefficient (undamped)

T_END  = 10.0   # total simulation time (s)
DT     = 0.001  # time step (s)
BETA   = 0.25   # Newmark parameter — average-acceleration (unconditionally stable)
GAMMA  = 0.5    # Newmark parameter

INPUT_VELOCITY_CASES = {
    'low':    -1.0,
    'medium': -2.0,
    'high':   -10.0,
}
SELECTED_DOFS = [0, 4, 9, 14, 19]   # 0-indexed, for plots
SAVE_DIR = os.path.join('Results_Linear_Chain', 'Newmark')


# ═══════════════════════════════════════════════════════════════════════════════
# System
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
                 t_end=T_END, dt=DT, beta=BETA, gamma=GAMMA):
    """
    Implicit Newmark-beta integrator for free vibration (F = 0).

    Returns
    -------
    t  : (n_steps,)       time vector
    x  : (n_steps, n_dof) displacements
    xt : (n_steps, n_dof) velocities
    """
    n       = M.shape[0]
    n_steps = int(round(t_end / dt)) + 1
    t       = np.linspace(0.0, t_end, n_steps)

    xh   = np.zeros((n, n_steps))
    xth  = np.zeros((n, n_steps))
    xtth = np.zeros((n, n_steps))

    xh[:,  0] = x0
    xth[:, 0] = xt0
    xtth[:, 0] = np.linalg.solve(M, -C @ xt0 - K @ x0)

    K_eff = M / (beta*dt**2) + gamma*C / (beta*dt) + K
    K_inv = np.linalg.inv(K_eff)

    for i in range(1, n_steps):
        F_eff = (
            M  @ (xh[:, i-1] / (beta*dt**2)
                  + xth[:, i-1] / (beta*dt)
                  + xtth[:, i-1] * (0.5/beta - 1.0))
            + C @ (gamma * xh[:, i-1] / (beta*dt)
                   - xth[:, i-1] * (1.0 - gamma/beta)
                   - dt * xtth[:, i-1] * (1.0 - gamma/(2.0*beta)))
        )
        xh[:, i]   = K_inv @ F_eff
        xtth[:, i] = ((xh[:, i] - xh[:, i-1]) / (beta*dt**2)
                      - xth[:, i-1] / (beta*dt)
                      - xtth[:, i-1] * (0.5/beta - 1.0))
        xth[:, i]  = (xth[:, i-1]
                      + dt * ((1.0 - gamma)*xtth[:, i-1] + gamma*xtth[:, i]))

    return t, xh.T, xth.T    # → (n_steps, n_dof)


def total_energy(x, xt, m=M_VAL, k=K_VAL):
    """x, xt : (n_time, n_dof)  →  E : (n_time,)"""
    Ek = 0.5 * m * np.sum(xt**2, axis=1)
    Ep = 0.5 * k  * np.sum((x[:, 1:] - x[:, :-1])**2, axis=1)
    return Ek + Ep


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Newmark-beta solver for a free-free linear chain.'
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

    print(f'N_DOF = {n_dof}  |  plotting DOFs: {[d+1 for d in selected_dofs]}')
    os.makedirs(SAVE_DIR, exist_ok=True)

    M, C, K = build_matrices(n=n_dof)

    for case, v_in in INPUT_VELOCITY_CASES.items():
        print(f'\n{"="*60}')
        print(f'  Case: {case}  |  v0 = {v_in:.2f} m/s'
              f'  |  E0 = {0.5*M_VAL*v_in**2:.4f} J')
        print(f'{"="*60}')

        x0, xt0 = left_velocity_ic(n=n_dof, v0=v_in)
        t, x, xt = newmark_beta(M, C, K, x0, xt0)
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
            ax.plot(t, x[:, di], 'k-', lw=1.2)
            ax.set_ylabel(f'DOF {di+1}  x (m)')
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'Newmark-β displacement  |  {n_dof} DOFs  |  '
                     f'{case.capitalize()}  v0={v_in:.1f} m/s', fontsize=12)
        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'newmark_displacement_{tag}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

        # ── velocity plot ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(len(selected_dofs), 1,
                                 figsize=(12, 3*len(selected_dofs)),
                                 sharex=True)
        for ax, di in zip(axes, selected_dofs):
            ax.plot(t, xt[:, di], 'C0-', lw=1.2)
            ax.set_ylabel(f'DOF {di+1}  ẋ (m/s)')
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'Newmark-β velocity  |  {n_dof} DOFs  |  '
                     f'{case.capitalize()}  v0={v_in:.1f} m/s', fontsize=12)
        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'newmark_velocity_{tag}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

        # ── energy plot ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, E, 'C2-', lw=1.4)
        ax.axhline(E[0], color='k', ls='--', lw=0.8, label=f'E0={E[0]:.4f} J')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Total Energy (J)')
        ax.set_title(f'Newmark-β energy  |  {n_dof} DOFs  |  '
                     f'{case.capitalize()}  v0={v_in:.1f} m/s')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(SAVE_DIR, f'newmark_energy_{tag}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

        # ── save data ──────────────────────────────────────────────────────────
        stem = f'newmark_{tag}'
        np.savez(os.path.join(SAVE_DIR, stem + '.npz'),
                 t=t, x=x, xt=xt, E=E,
                 v_in=np.array(v_in),
                 n_dof=np.array(n_dof))
        if _HAS_SAVEMAT:
            savemat(os.path.join(SAVE_DIR, stem + '.mat'),
                    {'t': t, 'x': x, 'xt': xt, 'E': E,
                     'v_in': np.array([[v_in]]),
                     'n_dof': np.array([[n_dof]])})
        print(f'  Data saved: {stem}.npz'
              + (' + .mat' if _HAS_SAVEMAT else ''))

    print(f'\nDone. Results in {SAVE_DIR}/')


if __name__ == '__main__':
    main()
