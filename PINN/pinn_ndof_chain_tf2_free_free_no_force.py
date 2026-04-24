"""
pinn_ndof_chain_tf2_free_free_no_force.py
----------------------------------------
PINN variant for a 20-DOF impact-damper chain with:
  1) FREE-FREE structural boundaries,
  2) no external harmonic force,
  3) wave launch from initial velocity at DOF 1.

This file reuses the original TF2 implementation and only changes the ODE
residual force term (F=0). It also provides matrix/IC helpers for the
free-free + initial-velocity scenario.
"""

import numpy as np
import tensorflow as tf

from pinn_ndof_chain_tf2 import (
    PIPNNs as _BasePIPNNs,
    find_impact_times,
    impact_velocity_update,
    propagate_ics,
    newmark_beta,
)


class PIPNNs(_BasePIPNNs):
    """Same PINN class, but with zero external force in the residual."""

    def _net_f(self, t):
        """ODE residual with no external force: M x_tt + C x_t + K x = 0."""
        x, x_t, x_tt = self._net_u(t)

        residual = (
            tf.transpose(tf.matmul(self.M_tf, x_tt, transpose_b=True))
            + tf.transpose(tf.matmul(self.C_tf, x_t, transpose_b=True))
            + tf.transpose(tf.matmul(self.K_tf, x, transpose_b=True))
        )
        return residual


def build_free_free_chain_matrices(n_dof=20, m_x=1.0, k=1.0, c=0.0):
    """
    Build M, C, K for a uniform nearest-neighbour chain with FREE-FREE ends.

    K (free-free):
      diag = [k, 2k, ..., 2k, k]
      offdiag = -k for neighbours.
    C uses the same topology with coefficient c.
    """
    n = int(n_dof)
    M = m_x * np.eye(n)
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


def make_left_velocity_ic(n_dof=20, x1_0=0.0, v1_0=1.0):
    """
    Initial conditions for wave launch from left end:
      x_1(0)=x1_0, x_i(0)=0 (i>1)
      xdot_1(0)=v1_0, xdot_i(0)=0 (i>1)
    """
    x0_total = np.zeros((1, n_dof), dtype=float)
    xt0_total = np.zeros((1, n_dof), dtype=float)
    x0_total[0, 0] = float(x1_0)
    xt0_total[0, 0] = float(v1_0)
    return x0_total, xt0_total
