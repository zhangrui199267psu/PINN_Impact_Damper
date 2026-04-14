# PINN Impact Damper Project Workspace

## Repository structure

- `reference/`: literature and supporting documents.
- `PINN/`: original non-parametric PINN implementation (Python + notebook).
- `Result/`: generated outputs, including `pinn_data.mat`.
- `parameter_PINN/`: new parametric workflow (frequency sweep) with Python script and notebook.

## Code analysis summary

The core solver is in `PINN/pinn_ndof_chain_tf2.py` and is organized in four blocks:

1. **PINN model class (`PIPNNs`)**
   - Builds a fully-connected network `t -> x(t)`.
   - Uses nested `tf.GradientTape` to obtain `x_t` and `x_tt`.
   - Loss combines initial-condition mismatch and ODE residual.
   - Optimizers: Adam + optional L-BFGS-B.

2. **Impact detection (`find_impact_times`)**
   - Evaluates separation gap over a scan grid.
   - Detects first valid closing/re-collision event.
   - Refines impact time with Brent root-finding.

3. **State update at impacts (`impact_velocity_update`, `propagate_ics`)**
   - Applies restitution/momentum update rule.
   - Builds next-segment initial conditions.

4. **Reference integrator (`newmark_beta`)**
   - Standard implicit Newmark-beta baseline solver.

## Goal (1): run and use `pinn_data.mat` to compute + draw dispersion curve

### Run original code
Use `PINN/pinn_ndof_chain_sim_tf2.ipynb` (or your existing execution flow) to produce output data and save to:

- `Result/pinn_data.mat`

### Dispersion post-processing notebook
A new notebook is provided:

- `PINN/dispersion_curve_from_pinn_data.ipynb`

This notebook:
- loads `Result/pinn_data.mat`,
- extracts displacement response matrix,
- computes frequency-wavenumber energy map (`f-k`) using FFT,
- plots a dispersion-style heat map.

## Goal (2): parametric PINN folder (varying excitation frequency)

A new folder is provided:

- `parameter_PINN/parametric_pinn_frequency.py`
- `parameter_PINN/parametric_pinn_frequency.ipynb`

### What it does
- Sweeps excitation frequencies (first trial parameter).
- For each frequency, configures forcing frequency input to PINN (`phi2`).
- Trains a segment model and predicts displacement response.
- Saves consolidated outputs to `Result/parametric_pinn_data.mat`.

### Example CLI run

```bash
python parameter_PINN/parametric_pinn_frequency.py \
  --freqs 20 30 40 50 \
  --tmax 1.0 \
  --npts 200 \
  --adam_iter 200
```

## Notes

- The parametric script is intentionally lightweight as a first-trial scaffold so you can quickly expand the parameter dimensions (e.g., amplitude, damping, gap).
- If your `.mat` fields differ, edit the field selection in the dispersion notebook.
