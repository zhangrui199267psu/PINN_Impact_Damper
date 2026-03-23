"""
parametric_pinn_multi_case.py
True Parametric PINN for the Impact Damper
-------------------------------------------
Trains ONE network per segment across N parameter cases simultaneously.
After a single training run, the network predicts x(t) for ANY new parameter
combination within the training bounds — no retraining required.

Usage:
    from parametric_pinn_multi_case import train_parametric_pinn

    predictor, models = train_parametric_pinn(
        n_cases=50,
        n_impacts=4,
        lb_params=[0.5, 0.1, 0.5, 0.0, 0.5, -2.0, -2.0, -1.0, -1.0],
        ub_params=[2.0, 0.6, 2.0, 0.2, 2.0,  2.0,  2.0,  1.0,  1.0],
        fixed_ic={'x0': 0.0, 'xt0': 0.0, 'y0': 0.0, 'yt0': -1.0},
        T_max_per_segment=[3.0, 5.0, 5.0, 5.0],
        nIter_per_segment=3000,
    )

    # Predict for any new parameter set — instant, no retraining:
    result = predictor.predict(mx=1.2, my=0.35, k=1.1, c=0.0, D=0.9)
"""

import warnings
import numpy as np
from scipy.optimize import brentq

from parametric_impact_pinn_tf1 import ParametricImpactPINN

# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def lhs_sample(n_cases, lb, ub, seed=42):
    """
    Latin Hypercube Sampling over parameter bounds.

    Parameters
    ----------
    n_cases : int
    lb, ub  : array-like of length n_params
    seed    : int

    Returns
    -------
    samples : (n_cases, n_params) float32 array
    """
    rng = np.random.RandomState(seed)
    lb = np.asarray(lb, dtype=np.float32)
    ub = np.asarray(ub, dtype=np.float32)
    n_p = len(lb)
    out = np.zeros((n_cases, n_p), dtype=np.float32)
    for i in range(n_p):
        perm = rng.permutation(n_cases)
        out[:, i] = lb[i] + (perm + rng.uniform(size=n_cases)) / n_cases * (ub[i] - lb[i])
    return out


# ---------------------------------------------------------------------------
# Root-finding: impact time for a trained segment model
# ---------------------------------------------------------------------------

def find_impact_time(model, mx, my, k, c, D, y0, yt0, x0, xt0,
                     T_max, n_scan=500, tol=1e-6):
    """
    Find the first t* in (0, T_max] where |x(t*; params) - y(t*)| = D.

    Scans the gap function over n_scan points, then uses Brent's method
    to refine the root. The network weights are frozen; only t* is searched.

    Returns
    -------
    t_impact : float
        Impact time. Returns T_max with a warning if no impact is found.
    """
    t_scan = np.linspace(1e-3, T_max, n_scan).reshape(-1, 1).astype(np.float32)
    x_scan, _, _ = model.predict(t_scan, x0=x0, xt0=xt0, y0=y0, yt0=yt0,
                                  mx=mx, my=my, k=k, c=c, D=D)
    x_scan = x_scan.flatten()
    y_scan = float(y0) + float(yt0) * t_scan.flatten()
    gap = np.abs(x_scan - y_scan) - float(D)

    sign_changes = np.where(np.diff(np.sign(gap)))[0]
    if len(sign_changes) == 0:
        warnings.warn(
            f"No impact detected within T_max={T_max}. "
            "Consider increasing T_max or checking parameter ranges."
        )
        return float(T_max)

    idx = sign_changes[0]
    ta = float(t_scan[idx, 0])
    tb = float(t_scan[idx + 1, 0])

    def _gap(t):
        t_arr = np.array([[t]], dtype=np.float32)
        xv, _, _ = model.predict(t_arr, x0=x0, xt0=xt0, y0=y0, yt0=yt0,
                                  mx=mx, my=my, k=k, c=c, D=D)
        yv = float(y0) + float(yt0) * t
        return abs(float(xv[0, 0]) - yv) - float(D)

    try:
        t_impact = brentq(_gap, ta, tb, xtol=tol)
    except ValueError:
        t_impact = ta

    return float(t_impact)


# ---------------------------------------------------------------------------
# Velocity update at impact
# ---------------------------------------------------------------------------

def _impact_update(mx, my, r, xt_minus, yt_minus):
    """
    Compute post-impact velocities via momentum conservation + restitution.

    Returns (xt_plus, yt_plus) as floats.
    """
    A = np.array([[mx, my], [1.0, -1.0]], dtype=np.float64)
    B = np.array([[mx, my], [-r,   r  ]], dtype=np.float64)
    v = np.linalg.solve(A, B @ np.array([[xt_minus], [yt_minus]], dtype=np.float64))
    return float(v[0, 0]), float(v[1, 0])


# ---------------------------------------------------------------------------
# Predictor: uses trained models for instant inference
# ---------------------------------------------------------------------------

class ParametricSequentialPredictor:
    """
    Wraps one trained ParametricImpactPINN per segment.

    At inference, calls find_impact_time() (root-finding on the frozen network)
    and chains segments via the impact update rule. No TF optimization runs.
    """

    def __init__(self, segment_models, r, T_max_per_segment):
        """
        Parameters
        ----------
        segment_models    : list of ParametricImpactPINN, one per impact segment
        r                 : float, coefficient of restitution
        T_max_per_segment : list of float, search window per segment
        """
        self.models = segment_models
        self.r = r
        self.T_max = T_max_per_segment

    def predict(self, mx, my, k, c, D,
                x0=0.0, xt0=0.0, y0=0.0, yt0=-1.0,
                num_points=500):
        """
        Predict full multi-impact response for the given parameter set.

        Parameters
        ----------
        mx, my, k, c, D : float   — system parameters
        x0, xt0, y0, yt0: float   — initial conditions
        num_points       : int    — points per segment in the output

        Returns
        -------
        dict with keys: 'time', 'x', 'xt', 'y', 'yt', 'impact_times'
        """
        all_t, all_x, all_xt, all_y, all_yt = [], [], [], [], []
        impact_times = []
        time_offset = 0.0

        cur_x0, cur_xt0 = float(x0), float(xt0)
        cur_y0, cur_yt0 = float(y0), float(yt0)

        for model, T_max in zip(self.models, self.T_max):
            t_imp = find_impact_time(
                model, mx, my, k, c, D,
                cur_y0, cur_yt0, cur_x0, cur_xt0, T_max)

            t_seg = np.linspace(0.0, t_imp, num_points).reshape(-1, 1).astype(np.float32)
            x_seg, xt_seg, _ = model.predict(
                t_seg,
                x0=cur_x0, xt0=cur_xt0, y0=cur_y0, yt0=cur_yt0,
                mx=mx, my=my, k=k, c=c, D=D)

            y_seg  = cur_y0 + cur_yt0 * t_seg
            yt_seg = np.full_like(t_seg, cur_yt0)

            all_t.append(t_seg + time_offset)
            all_x.append(x_seg)
            all_xt.append(xt_seg)
            all_y.append(y_seg)
            all_yt.append(yt_seg)
            impact_times.append(t_imp)
            time_offset += t_imp

            # State right at impact
            x1, xt1, _ = model.predict(
                np.array([[t_imp]], dtype=np.float32),
                x0=cur_x0, xt0=cur_xt0, y0=cur_y0, yt0=cur_yt0,
                mx=mx, my=my, k=k, c=c, D=D)

            y1   = cur_y0 + cur_yt0 * t_imp
            xt_plus, yt_plus = _impact_update(mx, my, self.r,
                                               float(xt1[0, 0]), cur_yt0)
            cur_x0, cur_xt0 = float(x1[0, 0]), xt_plus
            cur_y0, cur_yt0 = y1, yt_plus

        return {
            'time':         np.vstack(all_t),
            'x':            np.vstack(all_x),
            'xt':           np.vstack(all_xt),
            'y':            np.vstack(all_y),
            'yt':           np.vstack(all_yt),
            'impact_times': np.array(impact_times),
        }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_parametric_pinn(
    n_cases,
    n_impacts,
    lb_params,
    ub_params,
    r=1.0,
    fixed_ic=None,
    layers=None,
    T_max_per_segment=None,
    nIter_per_segment=5000,
    n_r=200,
    hyp_ini_para=0.5,
    hyp_ini_weight_loss=(1.0, 1.0, 10.0),
    optimizer_LB=True,
    seed=42,
):
    """
    Train one ParametricImpactPINN per impact segment, each on N cases at once.

    How it works
    ------------
    1. Sample n_cases parameter combinations via Latin Hypercube Sampling.
    2. For segment 1: create one ParametricImpactPINN with all n_cases as a
       batch. The network learns to satisfy the ODE + ICs + impact condition
       for every case simultaneously, sharing one set of weights.
    3. After training segment 1: find the impact time for each case via
       root-finding on the frozen network, then compute new ICs.
    4. Repeat for segments 2 … n_impacts.
    5. Return a ParametricSequentialPredictor that can predict the full
       response for any NEW parameter set without retraining.

    Parameters
    ----------
    n_cases : int
        Number of parameter combinations sampled for training (e.g. 50–200).
    n_impacts : int
        Number of impact segments.
    lb_params, ub_params : array-like of length 9
        Bounds for [mx, my, k, c, D, y0, yt0, x0, xt0].
    r : float
        Coefficient of restitution (fixed across all cases).
    fixed_ic : dict, optional
        Override sampled ICs. E.g. {'x0': 0.0, 'xt0': 0.0, 'y0': 0.0, 'yt0': -1.0}.
        Typically used to fix ICs and only vary physical params.
    layers : list of int
        Network architecture. Default [10, 64, 64, 64, 1].
    T_max_per_segment : list of float
        Max time window per segment (used for collocation + impact search).
    nIter_per_segment : int or list of int
        Adam iterations per segment.
    n_r : int
        Collocation points per case per segment.
    hyp_ini_para : float
        Initial guess for impact time (lambda_1).
    hyp_ini_weight_loss : tuple of 3 floats
        Weights for (IC loss, ODE loss, impact loss).
    optimizer_LB : bool
        If True, run L-BFGS-B after Adam (requires tf.contrib).
    seed : int
        Random seed for LHS.

    Returns
    -------
    predictor : ParametricSequentialPredictor
        Use predictor.predict(mx, my, k, c, D, ...) for instant inference.
    segment_models : list of ParametricImpactPINN
        Trained models, one per segment (for inspection / loss plots).
    """
    if layers is None:
        layers = [10, 64, 64, 64, 1]
    if T_max_per_segment is None:
        T_max_per_segment = [5.0] * n_impacts
    if isinstance(nIter_per_segment, int):
        nIter_per_segment = [nIter_per_segment] * n_impacts

    lb_params = np.asarray(lb_params, dtype=np.float32)
    ub_params = np.asarray(ub_params, dtype=np.float32)
    param_keys = ['mx', 'my', 'k', 'c', 'D', 'y0', 'yt0', 'x0', 'xt0']

    # ------------------------------------------------------------------
    # 1. Sample parameter cases
    # ------------------------------------------------------------------
    print(f"Sampling {n_cases} parameter cases via Latin Hypercube Sampling...")
    samples = lhs_sample(n_cases, lb_params, ub_params, seed=seed)
    cases = {k: samples[:, i:i+1] for i, k in enumerate(param_keys)}

    if fixed_ic is not None:
        for key, val in fixed_ic.items():
            if key in cases:
                cases[key] = np.full((n_cases, 1), float(val), dtype=np.float32)
                print(f"  Fixed IC: {key} = {val}")

    print(f"Parameter ranges used in training:")
    for i, k in enumerate(param_keys):
        print(f"  {k}: [{cases[k].min():.3f}, {cases[k].max():.3f}]")

    # ------------------------------------------------------------------
    # 2. Train one model per segment
    # ------------------------------------------------------------------
    segment_models = []

    for seg in range(n_impacts):
        T_max = T_max_per_segment[seg]
        nIter = nIter_per_segment[seg]
        print(f'\n{"="*60}')
        print(f'Segment {seg+1}/{n_impacts}  |  T_max={T_max}  |  '
              f'{n_cases} cases  |  {nIter} Adam iters')
        print(f'{"="*60}')

        t_r = np.linspace(0.0, T_max, n_r).reshape(-1, 1).astype(np.float32)
        t0  = np.zeros((n_cases, 1), dtype=np.float32)

        lb_full = np.concatenate([[0.0],   lb_params]).astype(np.float32)
        ub_full = np.concatenate([[T_max], ub_params]).astype(np.float32)

        model = ParametricImpactPINN(
            lb=lb_full,
            ub=ub_full,
            t0=t0,
            t_r=t_r,
            x0=cases['x0'],
            xt0=cases['xt0'],
            y0=cases['y0'],
            yt0=cases['yt0'],
            mx=cases['mx'],
            my=cases['my'],
            k=cases['k'],
            c=cases['c'],
            D=cases['D'],
            layers=layers,
            hyp_ini_weight_loss=hyp_ini_weight_loss,
            hyp_ini_para=hyp_ini_para,
            optimizer_LB=optimizer_LB,
        )

        model.train(nIter=nIter, optimizer_LB=optimizer_LB,
                    print_every=max(1, nIter // 10))
        segment_models.append(model)

        # ------------------------------------------------------------------
        # 3. Propagate each case through the impact to get ICs for seg+1
        # ------------------------------------------------------------------
        if seg < n_impacts - 1:
            print(f'\nPropagating {n_cases} cases to compute ICs for segment {seg+2}...')
            new_x0  = np.zeros((n_cases, 1), dtype=np.float32)
            new_xt0 = np.zeros((n_cases, 1), dtype=np.float32)
            new_y0  = np.zeros((n_cases, 1), dtype=np.float32)
            new_yt0 = np.zeros((n_cases, 1), dtype=np.float32)

            for i in range(n_cases):
                p = {k: float(cases[k][i, 0]) for k in param_keys}

                t_imp = find_impact_time(
                    model,
                    p['mx'], p['my'], p['k'], p['c'], p['D'],
                    p['y0'], p['yt0'], p['x0'], p['xt0'], T_max)

                x1, xt1, _ = model.predict(
                    np.array([[t_imp]], dtype=np.float32),
                    x0=p['x0'], xt0=p['xt0'], y0=p['y0'], yt0=p['yt0'],
                    mx=p['mx'], my=p['my'], k=p['k'], c=p['c'], D=p['D'])

                y1 = p['y0'] + p['yt0'] * t_imp
                xt_plus, yt_plus = _impact_update(
                    p['mx'], p['my'], r, float(xt1[0, 0]), p['yt0'])

                new_x0[i, 0]  = float(x1[0, 0])
                new_xt0[i, 0] = xt_plus
                new_y0[i, 0]  = y1
                new_yt0[i, 0] = yt_plus

            # Physical parameters stay the same; only ICs update
            cases['x0']  = new_x0
            cases['xt0'] = new_xt0
            cases['y0']  = new_y0
            cases['yt0'] = new_yt0

    predictor = ParametricSequentialPredictor(segment_models, r, T_max_per_segment)
    print(f'\nTraining complete. Use predictor.predict(mx, my, k, c, D, ...) '
          f'for instant inference on any new parameter set.')
    return predictor, segment_models
