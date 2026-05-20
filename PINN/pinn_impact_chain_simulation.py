"""
pinn_impact_chain_simulation.py
--------------------------------
End-to-end simulation script for the 20-DOF free-free impact-damper chain.

Converted from pinn_impact_chain_simulation.ipynb so the full workflow can be
run directly from the terminal:

    python pinn_impact_chain_simulation.py

Workflow
--------
1.  Build system matrices and initial conditions
2.  Compute Newmark-beta reference trajectories for all three energy cases
3.  Run the multi-impact PINN driver for all three cases
4.  Plot diagnostics (relative displacement, velocity, energy, impulse forces)
5.  Compute and plot dispersion maps
6.  Save all results (NPZ + MAT + CSV)

All figures are written to disk (Results_free_free/); no interactive display
is required.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from pinn_impact_chain_solver import (
    PIPNNs,
    build_free_free_chain_matrices,
    make_left_velocity_ic,
    find_impact_times,
    propagate_ics,
    newmark_beta,
)

# optional MATLAB export
try:
    from scipy.io import savemat
    _HAS_SAVEMAT = True
except ImportError:
    _HAS_SAVEMAT = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1234)
tf.random.set_seed(1234)
print('TF version:', tf.__version__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1)  Parameters
# ═══════════════════════════════════════════════════════════════════════════════

n_dof = 20
m_x   = 1.0
m_y   = 0.3
k     = 1.0
c     = 0.0
D     = 1.0
r     = 1.0

# no external forcing
phi1 = 0.0
phi2 = 0.0

input_velocity_cases = {
    'low':    -1.0,
    'medium': -2.0,
    'high':   -10.0,
}

# PINN / training controls
x1_0   = 0.0
layers = [1, 64, n_dof]
hyp_ini_weight_loss  = np.array([1.0, 1.0])
optimizer_LB_value   = True
T_segment            = 1.0
n_t_segment          = 100
nIter_per_segment    = 1000
t_end_target         = 10.0
tau_contact          = 1e-3

# Newmark reference controls
T_ref  = 10.0
dt_ref = 0.001

save_dir = 'Results_free_free'


# ═══════════════════════════════════════════════════════════════════════════════
# 2)  Build system matrices
# ═══════════════════════════════════════════════════════════════════════════════

M_total, C_total, K_total = build_free_free_chain_matrices(
    n_dof=n_dof, m_x=m_x, k=k, c=c
)

y0  = np.zeros(n_dof)
yt0 = np.zeros(n_dof)

A = np.array([[m_x, m_y],
              [1.0, -1.0]])
B = np.array([[m_x, m_y],
              [-r,   r  ]])
A_inv_B = np.linalg.inv(A) @ B

print('M/K/C shape:', M_total.shape, K_total.shape, C_total.shape)


# ═══════════════════════════════════════════════════════════════════════════════
# 3)  Newmark reference — early-time displacement plots
# ═══════════════════════════════════════════════════════════════════════════════

def run_newmark_reference():
    num_ref  = int(T_ref / dt_ref) + 1
    t_ref    = np.linspace(0.0, T_ref, num_ref)
    F_ref    = np.zeros((n_dof, num_ref))
    sel_dofs = [0, 4, 9, 14, 19]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for ax, (case_name, v1_0_case) in zip(axes, input_velocity_cases.items()):
        x0_case, xt0_case = make_left_velocity_ic(
            n_dof=n_dof, x1_0=x1_0, v1_0=v1_0_case
        )
        u_ref, _, _ = newmark_beta(
            M_total, C_total, K_total, F_ref,
            dt=dt_ref, n_steps=num_ref, n_dof=n_dof,
            x0=x0_case[0], xt0=xt0_case[0],
        )
        for i in sel_dofs:
            ax.plot(t_ref, u_ref[i, :], lw=1.2, label=f'DOF {i+1}')
        ax.set_title(f'{case_name.capitalize()} — $v_1(0)$ = {v1_0_case}')
        ax.set_ylabel('Displacement')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes[0].legend(ncol=5, fontsize=8)
    fig.suptitle('Newmark-β reference (no impactors)', fontsize=12)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'newmark_reference.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 4)  Multi-impact PINN driver
# ═══════════════════════════════════════════════════════════════════════════════

def segment_total_energy(x_seg, xt_seg, yt_seg):
    Ek_x    = 0.5 * m_x * np.sum(xt_seg**2, axis=1)
    Ek_y    = 0.5 * m_y * np.sum(yt_seg**2) * np.ones(x_seg.shape[0])
    spring_rel = x_seg[:, 1:] - x_seg[:, :-1]
    Ep_s    = 0.5 * k * np.sum(spring_rel**2, axis=1)
    return Ek_x + Ek_y + Ep_s


def run_free_free_pinn_multiimpact(
    t_end_target,
    T_segment,
    n_t_segment,
    nIter_per_segment,
    tau_contact=1e-3,
    x0_init=None,
    xt0_init=None,
    case_name='case',
):
    lb = np.array([0.0])
    ub = np.array([float(T_segment)])

    if x0_init is None or xt0_init is None:
        raise ValueError('x0_init and xt0_init must be provided.')

    x0_cur  = x0_init.copy()
    xt0_cur = xt0_init.copy()
    y0_cur  = y0.copy()
    yt0_cur = yt0.copy()

    phi_cur    = np.array([[0.0]])
    t_global   = 0.0

    t_hist, x_hist, xt_hist, y_hist, yt_hist = [], [], [], [], []
    E_hist              = []
    impact_log          = []
    segment_train_times = []
    n_impacts_found     = 0
    t_train_total_start = time.perf_counter()
    seg_id              = 0

    while t_global < float(t_end_target):
        seg_id += 1
        t_seg = np.linspace(
            0.0, float(T_segment), int(n_t_segment)
        ).reshape(-1, 1)

        seg_train_start = time.perf_counter()
        model = PIPNNs(
            lb, ub,
            np.array([[0.0]]), t_seg,
            x0_cur, xt0_cur,
            y0_cur, yt0_cur,
            M_total, K_total, D, n_dof,
            phi_cur, phi1, phi2,
            layers,
            hyp_ini_weight_loss,
            C=C_total,
            optimizer_LB=optimizer_LB_value,
        )
        model.train(nIter=int(nIter_per_segment),
                    optimizer_LB=optimizer_LB_value)
        seg_train_time = time.perf_counter() - seg_train_start
        segment_train_times.append(seg_train_time)

        t_impacts, hit = find_impact_times(
            model, y0_cur, yt0_cur, D, float(T_segment)
        )

        if np.any(hit):
            j        = int(np.argmin(t_impacts))
            t_event  = float(t_impacts[j])
            has_impact = np.isfinite(t_event) and (0.0 < t_event < float(T_segment))
        else:
            j          = None
            t_event    = float(T_segment)
            has_impact = False

        if not np.isfinite(t_event) or t_event <= 0.0:
            print(f'[{case_name}] [stop] invalid t_event={t_event} at seg {seg_id}')
            break

        t_remaining = float(t_end_target) - t_global
        t_event     = min(t_event, t_remaining)

        t_local = np.linspace(0.0, t_event, int(n_t_segment) + 1).reshape(-1, 1)
        x_local, xt_local, _ = model.predict(t_local)
        y_local  = y0_cur + np.outer(t_local.flatten(), yt0_cur)
        E_local  = segment_total_energy(x_local, xt_local, yt0_cur)

        t_global_local = t_global + t_local.flatten()
        yt_local       = np.tile(yt0_cur, (len(t_local), 1))

        if len(t_hist) > 0:
            t_global_local = t_global_local[1:]
            x_local  = x_local[1:]
            xt_local = xt_local[1:]
            y_local  = y_local[1:]
            yt_local = yt_local[1:]
            E_local  = E_local[1:]

        t_hist.append(t_global_local)
        x_hist.append(x_local)
        xt_hist.append(xt_local)
        y_hist.append(y_local)
        yt_hist.append(yt_local)
        E_hist.append(E_local)

        x_hit, xt_hit, _ = model.predict(
            np.array([[t_event]], dtype=np.float32)
        )

        can_impact = has_impact and (t_event < t_remaining + 1e-14)
        if can_impact:
            xt_minus = float(xt_hit[0, j])
            yt_minus = float(yt0_cur[j])
            V0 = np.array([[xt_minus], [yt_minus]])
            V1 = A_inv_B @ V0
            xt_plus = float(V1[0])
            yt_plus = float(V1[1])

            Jx   = m_x * (xt_plus - xt_minus)
            Jmag = abs(Jx)
            Feq  = Jmag / max(float(tau_contact), 1e-12)

            x0_next, xt0_next, y0_next, yt0_next = propagate_ics(
                model, t_event,
                x_hit, xt_hit,
                y0_cur, yt0_cur,
                j,
                m_x * np.ones(n_dof), m_y * np.ones(n_dof), r,
                A_inv_B,
            )

            n_impacts_found += 1
            impact_log.append({
                'impact_id':        n_impacts_found,
                'segment_id':       seg_id,
                'dof':              j + 1,
                't_local':          t_event,
                't_global':         t_global + t_event,
                'xt_minus':         xt_minus,
                'yt_minus':         yt_minus,
                'xt_plus':          xt_plus,
                'yt_plus':          yt_plus,
                'impulse_x':        Jx,
                'impulse_abs':      Jmag,
                'force_eq':         Feq,
                'segment_train_time_s': seg_train_time,
            })
            print(
                f'[{case_name}] seg {seg_id:03d}: '
                f'impact #{n_impacts_found:02d}  DOF {j+1:02d}  '
                f't_global={t_global + t_event:.6f}  train={seg_train_time:.2f}s'
            )
        else:
            x0_next  = x_hit.copy()
            xt0_next = xt_hit.copy()
            y0_next  = y0_cur  + yt0_cur * t_event
            yt0_next = yt0_cur.copy()
            print(
                f'[{case_name}] seg {seg_id:03d}: '
                f'no-impact to t={t_global + t_event:.6f}s  '
                f'train={seg_train_time:.2f}s'
            )

        t_global += t_event
        phi_cur   = phi_cur + t_event
        x0_cur, xt0_cur, y0_cur, yt0_cur = (
            x0_next, xt0_next, y0_next, yt0_next
        )

    if len(t_hist) == 0:
        return None

    total_train_time_s = time.perf_counter() - t_train_total_start
    return {
        't':                    np.concatenate(t_hist),
        'x':                    np.vstack(x_hist),
        'xt':                   np.vstack(xt_hist),
        'y':                    np.vstack(y_hist),
        'yt':                   np.vstack(yt_hist),
        'energy_total':         np.concatenate(E_hist),
        'impacts':              impact_log,
        'n_impacts':            n_impacts_found,
        'n_segments':           seg_id,
        'segment_train_times_s': np.asarray(segment_train_times),
        'total_train_time_s':   total_train_time_s,
        't_end_target_s':       float(t_end_target),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5)  Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_diagnostics(batch_results, out_dir):
    case_names = list(batch_results.keys())
    fig, axes  = plt.subplots(5, len(case_names),
                               figsize=(6 * len(case_names), 14),
                               sharex='col')
    if len(case_names) == 1:
        axes = axes[:, np.newaxis]

    for col, case_name in enumerate(case_names):
        res = batch_results[case_name]
        t   = res['t']
        x   = res['x']
        xt  = res['xt']
        y   = res['y']
        yt  = res['yt']
        E   = res['energy_total']

        t_imp = np.array([ev['t_global'] for ev in res['impacts']]) \
                if res['impacts'] else np.array([])
        F_imp = np.array([ev['force_eq'] for ev in res['impacts']]) \
                if res['impacts'] else np.array([])

        rel = x - y

        # relative displacement
        ax = axes[0, col]
        for i in range(n_dof):
            ax.plot(t, rel[:, i], lw=0.8, alpha=0.7)
        ax.axhline( D, color='r', ls='--', lw=1.0)
        ax.axhline(-D, color='r', ls='--', lw=1.0)
        for tt in t_imp: ax.axvline(tt, color='k', lw=0.5, ls=':', alpha=0.3)
        ax.set_title(case_name.capitalize())
        ax.set_ylabel(r'$x_i - y_i$')
        ax.grid(alpha=0.25)

        # primary velocity
        ax = axes[1, col]
        for i in range(n_dof):
            ax.plot(t, xt[:, i], lw=0.8, alpha=0.7)
        for tt in t_imp: ax.axvline(tt, color='k', lw=0.5, ls=':', alpha=0.3)
        ax.set_ylabel(r'$\dot{x}_i$')
        ax.grid(alpha=0.25)

        # impactor velocity
        ax = axes[2, col]
        for i in range(n_dof):
            ax.plot(t, yt[:, i], lw=0.8, alpha=0.7)
        for tt in t_imp: ax.axvline(tt, color='k', lw=0.5, ls=':', alpha=0.3)
        ax.set_ylabel(r'$\dot{y}_i$')
        ax.grid(alpha=0.25)

        # equivalent impulse force
        ax = axes[3, col]
        if len(t_imp):
            ax.stem(t_imp, F_imp,
                    basefmt=' ', linefmt='C3-', markerfmt='C3o')
        ax.set_ylabel('F_eq (N)')
        ax.grid(alpha=0.25)

        # total energy
        ax = axes[4, col]
        ax.plot(t, E, lw=1.3, color='C2')
        for tt in t_imp: ax.axvline(tt, color='k', lw=0.5, ls=':', alpha=0.2)
        ax.set_ylabel('Energy (J)')
        ax.set_xlabel('Time (s)')
        ax.grid(alpha=0.25)

    fig.suptitle('PINN impact-damper chain — diagnostics', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'pinn_diagnostics.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 6)  Dispersion curves
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dispersion_map(x_tn, dt, d=1.0):
    """
    x_tn : (n_time, n_dof) response matrix
    Returns k (rad/m), omega (rad/s), S (normalised |FFT|²)
    """
    x_tn = np.asarray(x_tn, dtype=float)
    n_t, n_x = x_tn.shape

    x0 = x_tn - np.mean(x_tn, axis=0, keepdims=True)
    wt = np.hanning(n_t)[:, None]
    wx = np.hanning(n_x)[None, :]
    xw = x0 * wt * wx

    F_full   = np.fft.fft2(xw)
    S_full   = np.abs(F_full)

    omega_raw = 2.0 * np.pi * np.fft.fftfreq(n_t, d=dt)
    k_raw     = 2.0 * np.pi * np.fft.fftfreq(n_x, d=d)

    omega_mask = omega_raw >= 0.0
    k_mask     = (k_raw >= 0.0) & (k_raw <= np.pi / d + 1e-12)

    omega = omega_raw[omega_mask]
    k     = k_raw[k_mask]
    S     = S_full[np.ix_(omega_mask, k_mask)]
    if np.max(S) > 0:
        S = S / np.max(S)

    return k, omega, S


def plot_dispersion(batch_results, out_dir, omega_max=None):
    case_names = list(batch_results.keys())
    fig, axes  = plt.subplots(1, len(case_names),
                               figsize=(5 * len(case_names), 4), sharey=True)
    if len(case_names) == 1:
        axes = [axes]

    first = next(iter(batch_results.values()))
    dt_est = float(np.mean(np.diff(first['t'][:min(len(first['t']), 200)])))

    for ax, case_name in zip(axes, case_names):
        res  = batch_results[case_name]
        k, omega, S = compute_dispersion_map(res['x'], dt=dt_est, d=1.0)
        kpi  = k / (np.pi / 1.0)
        pcm  = ax.pcolormesh(kpi, omega, S, shading='auto', cmap='magma')
        ax.set_title(f"{case_name} (v0={res['v_input']:.2f} m/s)")
        ax.set_xlabel('k / π')
        ax.set_xlim(0.0, 1.0)
        if omega_max is not None:
            ax.set_ylim(0.0, omega_max)
        ax.grid(alpha=0.2)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label='|FFT| (norm.)')

    axes[0].set_ylabel('ω (rad/s)')
    fig.suptitle('Dispersion maps — 3 velocity cases', y=1.02, fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'dispersion_maps.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 7)  Save results
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(batch_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []

    for case_name, res in batch_results.items():
        stem     = f'pinn_free_free_20dof_{case_name}'
        npz_path = os.path.join(out_dir, stem + '.npz')

        np.savez(
            npz_path,
            t=res['t'], x=res['x'], xt=res['xt'],
            y=res['y'], yt=res['yt'],
            energy_total=res['energy_total'],
            impacts=np.array(res['impacts'], dtype=object),
            segment_train_times_s=res['segment_train_times_s'],
            total_train_time_s=res['total_train_time_s'],
            v_input=res['v_input'],
            E0=res['E0'],
        )

        if _HAS_SAVEMAT:
            savemat(
                os.path.join(out_dir, stem + '.mat'),
                {
                    't_total':        res['t'],
                    'x_PINN_total':   res['x'],
                    'xt_PINN_total':  res['xt'],
                    'y_total':        res['y'],
                    'yt_total':       res['yt'],
                    'energy_total':   res['energy_total'],
                    'v_input':        np.array([[res['v_input']]]),
                    'E0':             np.array([[res['E0']]]),
                    'n_impacts':      np.array([[res['n_impacts']]]),
                    'n_segments':     np.array([[res['n_segments']]]),
                    'train_time_total_s': np.array([[res['total_train_time_s']]]),
                }
            )

        summary_rows.append((
            case_name, res['v_input'], res['E0'],
            res['n_impacts'], res['n_segments'],
            res['t'][-1], res['total_train_time_s'],
        ))

    csv_path = os.path.join(out_dir, 'batch_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('case,v_input,E0,n_impacts,n_segments,t_end_s,total_train_time_s\n')
        for row in summary_rows:
            f.write(','.join(str(v) for v in row) + '\n')

    print(f'\nSaved all results to: {out_dir}/')
    print('MATLAB export:', 'yes' if _HAS_SAVEMAT else 'no (scipy not found)')
    for row in summary_rows:
        print(
            f"  case={row[0]:>6s}  v={row[1]:>7.3f}  E0={row[2]:>9.4f}  "
            f"impacts={row[3]:>3d}  segments={row[4]:>3d}  "
            f"t_end={row[5]:>6.2f}s  train={row[6]:>8.2f}s"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(save_dir, exist_ok=True)

    # step 3: Newmark reference plots
    run_newmark_reference()

    # step 4: run all PINN cases
    batch_results = {}
    for case_name, v_in in input_velocity_cases.items():
        x0_case, xt0_case = make_left_velocity_ic(
            n_dof=n_dof, x1_0=x1_0, v1_0=v_in
        )
        E0_case = 0.5 * m_x * v_in**2

        print('\n' + '='*72)
        print(f'Running case: {case_name}  |  v1_0={v_in:.3f} m/s  |  E0={E0_case:.4f} J')
        print(f'Target time: {t_end_target:.2f} s')
        print('='*72)

        res = run_free_free_pinn_multiimpact(
            t_end_target=t_end_target,
            T_segment=T_segment,
            n_t_segment=n_t_segment,
            nIter_per_segment=nIter_per_segment,
            tau_contact=tau_contact,
            x0_init=x0_case,
            xt0_init=xt0_case,
            case_name=case_name,
        )

        if res is None:
            print(f'Case {case_name} failed: no trajectories generated.')
            continue

        res['case_name'] = case_name
        res['v_input']   = v_in
        res['E0']        = E0_case
        batch_results[case_name] = res

        print(
            f'[{case_name}]  impacts={res["n_impacts"]}  '
            f'segments={res["n_segments"]}  '
            f't_end={res["t"][-1]:.3f}s  '
            f'train_total={res["total_train_time_s"]:.2f}s'
        )

    if not batch_results:
        raise RuntimeError('No case completed successfully.')

    # step 5: diagnostic plots
    plot_diagnostics(batch_results, save_dir)

    # step 6: dispersion curves
    plot_dispersion(batch_results, save_dir, omega_max=2.0)

    # step 7: save
    save_results(batch_results, save_dir)


if __name__ == '__main__':
    main()
