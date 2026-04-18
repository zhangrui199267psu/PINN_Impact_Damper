# Parametric PINN (single model over full 50 impacts)

This folder now keeps the **original ideas/settings** from:
- `PINN/pinn_ndof_chain_sim_tf2.ipynb`
- `PINN/pinn_ndof_chain_tf2.py`

including original physical settings and training workflow (Adam + L-BFGS).

## What is implemented

A two-stage workflow:

1. **Legacy trajectory generation (original PINN logic):**
   - For sampled `(phi1, phi2)` pairs, run the original segment-by-segment PINN (`PIPNNs`) for 50 impacts.
   - Keep original defaults (`n_dof=20`, `m_x=1.0`, `m_y=0.3`, `k=1.0`, `D=1.0`, segment training with Adam + optional L-BFGS, etc.).
2. **One parametric network for full horizon:**
   - Train one model mapping `(t, phi1, phi2) -> x(t)` over all sampled cases (aligned with original PINN output convention).
   - For unseen in-range parameters, predict full-horizon `x(t)` directly and extract impact times with a two-phase root-finding strategy aligned with `pinn_ndof_chain_tf2.py`.

## Files

- `parametric_pinn_50_impacts.py`
  - Contains:
    - `LegacySimConfig` (original simulation/training defaults)
    - `ParametricDataConfig`
    - `ParametricModelConfig`
    - `LegacyFullHorizonGenerator` (uses original `PIPNNs`, `find_impact_times`, `propagate_ics`)
    - `ParametricFullHorizonPINN` (single network with Adam + L-BFGS, plus phase-2 impact-time root finding)
- `parametric_pinn_50_impacts.ipynb`
  - Notebook runner for the full pipeline.

## Quick start

```bash
python "parametric PINN/parametric_pinn_50_impacts.py"
```

Or run interactively via:

- `parametric PINN/parametric_pinn_50_impacts.ipynb`
