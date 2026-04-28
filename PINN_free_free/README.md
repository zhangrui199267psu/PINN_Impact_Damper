# PINN_free_free

This folder contains free-free impact-chain workflows for 20 DOFs.

## Files

- `pinn_ndof_chain_tf2_free_free_no_force.py`  
  Standalone TF2 PINN utilities (free-free, no external forcing).

- `pinn_ndof_chain_sim_tf2_free_free_icv.ipynb`  
  Main simulation notebook for multiple initial-velocity cases (low / medium / high), including:
  - 20-second simulations,
  - impact detection and impulse statistics,
  - velocity/relative-displacement/energy plots,
  - 3-case dispersion-map plotting,
  - per-case result export.

- `pinn_ndof_chain_parametric_tf2.py`  
  Parametric velocity-conditioned surrogate workflow:
  - event-based data generation,
  - response surrogate: `(t, v0) -> x(t, v0)`,
  - impact-time surrogate: `v0 -> first-N impact times`,
  - fast inference for unseen velocities inside the training range.

---

## Quick Start

### 1) Notebook workflow

Open and run:

```bash
jupyter notebook PINN_free_free/pinn_ndof_chain_sim_tf2_free_free_icv.ipynb
```

Expected workflow in notebook:
1. Set system/training parameters.
2. Run low/medium/high velocity cases.
3. Plot diagnostics and dispersion maps.
4. Save outputs to `Result_free_free/`.

### 2) Parametric surrogate workflow

Run:

```bash
python PINN_free_free/pinn_ndof_chain_parametric_tf2.py
```

This will:
1. Generate training data from event simulations.
2. Train response and impact-time surrogate models.
3. Predict response and impact times for a test velocity.
4. Save model artifacts in `PINN_free_free/parametric_model_artifacts/`.

---

## Outputs

### Notebook output folder: `Result_free_free/`

Per case (`low`, `medium`, `high`):
- `pinn_free_free_20dof_<case>.npz`
- `pinn_free_free_20dof_<case>.mat` (if `scipy.io.savemat` available)

And summary:
- `batch_summary.csv`

### Parametric model output folder: `PINN_free_free/parametric_model_artifacts/`

- `response_model.keras`
- `impact_time_model.keras`
- `meta.npz`

---

## Notes

- The notebook simulation stopping criterion is **target physical time** (default 20 s), not fixed impact count.
- For best parametric generalization, choose training velocities that cover your intended inference range.
- If you observe over/under-smoothing in surrogate predictions, increase velocity-grid density and training epochs.
