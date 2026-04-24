"""
free_free_20dof_initial_velocity.py
-----------------------------------
20-DOF linear chain, FREE-FREE boundary conditions, no external force.

Scenario requested:
- original: left fixed, right free
- new:      left free, right free
- excitation: only initial condition at the left-end DOF
              (equivalent to an external mesh initial displacement/velocity)
- no harmonic external force during time marching

Model:
    M x_ddot + C x_dot + K x = 0

with nearest-neighbour springs and free-free boundaries.
"""

from __future__ import annotations

import numpy as np


def build_chain_mck_free_free(
    n_dof: int = 20,
    m: float = 1.0,
    k: float = 1.0,
    c: float = 0.0,
):
    """
    Build M, C, K for a uniform 1D chain with FREE-FREE boundaries.

    K structure (free-free):
      diag:     [k, 2k, 2k, ..., 2k, k]
      offdiag:  -k on +/-1 diagonals

    C here uses the same topology with coefficient c.
    """
    n = int(n_dof)
    M = m * np.eye(n)

    K = np.zeros((n, n), dtype=float)
    C = np.zeros((n, n), dtype=float)

    for i in range(n):
        if i > 0:
            K[i, i] += k
            K[i, i - 1] -= k
            C[i, i] += c
            C[i, i - 1] -= c
        if i < n - 1:
            K[i, i] += k
            K[i, i + 1] -= k
            C[i, i] += c
            C[i, i + 1] -= c

    return M, C, K


def simulate_free_free_initial_condition(
    n_dof: int = 20,
    m: float = 1.0,
    k: float = 1.0,
    c: float = 0.0,
    t_end: float = 20.0,
    dt: float = 1e-3,
    x_mesh0: float = 0.0,
    v_mesh0: float = 1.0,
    beta: float = 0.25,
    gamma: float = 0.5,
):
    """
    Time-march a free-free chain under initial-only excitation.

    Initial condition mapping:
      x0[0] = x_mesh0
      v0[0] = v_mesh0
      all other DOFs start at 0.

    Returns
    -------
    t : (nt,) time array
    x : (nt, n_dof) displacement history
    v : (nt, n_dof) velocity history
    a : (nt, n_dof) acceleration history
    """
    M, C, K = build_chain_mck_free_free(n_dof=n_dof, m=m, k=k, c=c)

    t = np.arange(0.0, t_end + 1e-15, dt)
    nt = t.size

    x = np.zeros((nt, n_dof), dtype=float)
    v = np.zeros((nt, n_dof), dtype=float)
    a = np.zeros((nt, n_dof), dtype=float)

    # Initial conditions (left-end DOF only)
    x[0, 0] = float(x_mesh0)
    v[0, 0] = float(v_mesh0)

    # No external force
    f0 = np.zeros(n_dof, dtype=float)
    a[0] = np.linalg.solve(M, f0 - C @ v[0] - K @ x[0])

    # Newmark constants
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    K_eff = K + a0 * M + a1 * C

    for i in range(nt - 1):
        # zero forcing at all times
        f_next = np.zeros(n_dof, dtype=float)

        rhs = (
            f_next
            + M @ (a0 * x[i] + a2 * v[i] + a3 * a[i])
            + C @ (a1 * x[i] + a4 * v[i] + a5 * a[i])
        )

        x[i + 1] = np.linalg.solve(K_eff, rhs)
        a[i + 1] = a0 * (x[i + 1] - x[i]) - a2 * v[i] - a3 * a[i]
        v[i + 1] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a[i + 1])

    return t, x, v, a


if __name__ == "__main__":
    # Example usage for requested case
    t, x, v, a = simulate_free_free_initial_condition(
        n_dof=20,
        m=1.0,
        k=1.0,
        c=0.02,
        t_end=15.0,
        dt=1e-3,
        x_mesh0=0.0,   # external mesh initial displacement
        v_mesh0=1.0,   # external mesh initial velocity (wave launch)
    )

    print("Simulation finished")
    print("t shape:", t.shape)
    print("x shape:", x.shape)
    print("Left DOF max |x|:", np.max(np.abs(x[:, 0])))
    print("Right DOF max |x|:", np.max(np.abs(x[:, -1])))
