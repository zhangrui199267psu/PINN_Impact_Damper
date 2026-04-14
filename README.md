# PINN Impact Damper Workflow

This repository is now organized for your **Goal 1**:

1. Run the PINN simulation to generate response data (`pinn_data.mat`).
2. Use that data to compute and plot dispersion curves and verify the dispersion code.

## Folder structure

- `reference/`
  - Research paper / background document.
- `PINN/`
  - PINN simulation code (Python + notebook).
- `Dispersion/`
  - Dispersion analysis code (theory + FFT-from-MAT workflow, Python + notebooks).
- `Result/`
  - Output data files, including `pinn_data.mat`.

## Code roles (quick analysis)

### PINN stage
- `PINN/pinn_ndof_chain_tf2.py`
  - TensorFlow 2 PINN solver for the nonlinear n-DOF impact-damper chain.
  - Handles segment-by-segment dynamics with impact-time detection and IC propagation.
- `PINN/pinn_ndof_chain_sim_tf2.ipynb`
  - Notebook workflow to configure parameters, train/run the PINN simulation, and assemble global response.

### Dispersion stage
- `Dispersion/pinn_dispersion_from_mat.py`
  - Loads PINN output from `.mat`, resamples to uniform time grid, performs 2D FFT, and extracts/plots dispersion information.
- `Dispersion/pinn_dispersion_from_mat.ipynb`
  - Notebook version for interactive dispersion processing and plotting.
- `Dispersion/dispersion_theory.py`
  - Analytical reference formulas (monoatomic and mass-in-mass dispersion, band-gap edges, group velocity).
- `Dispersion/dispersion_tutorial.ipynb`
  - Tutorial-style derivations/plots for linear dispersion concepts.

## Goal 1: recommended run order

## Step A — Generate/refresh PINN response data
Open and run:

- `PINN/pinn_ndof_chain_sim_tf2.ipynb`

At the end of the notebook, save to:

- `Result/pinn_data.mat`

If needed from Python, use `save_pinn_results(...)` from `Dispersion/pinn_dispersion_from_mat.py`.

## Step B — Compute and verify dispersion from PINN data
Open and run either:

- `Dispersion/pinn_dispersion_from_mat.ipynb`

or script workflow with:

- `Dispersion/pinn_dispersion_from_mat.py`

Set data path to:

- `Result/pinn_data.mat`

Then compare FFT-derived ridge / heatmap behavior against analytical trends from:

- `Dispersion/dispersion_theory.py`

## Notes for double-checking dispersion code

When validating, check the following:

1. **Uniform resampling quality** (`resample_to_uniform`):
   - Verify time-step `dt` resolves your highest expected harmonic.
2. **Transient removal** (`skip_transient` in 2D FFT):
   - Confirm startup transients are removed before FFT.
3. **Ridge extraction stability**:
   - Check sensitivity to `omega_min` and spectral dynamic range.
4. **Theory consistency**:
   - For near-linear/small-amplitude runs, the dominant band should align with analytical branch trends.

## Next step (Goal 2 preview)

You asked for a future `parameter_PINN/` folder with new `.py` + `.ipynb` for a parametric PINN (first trial: varying excitation frequencies). This README is prepared for Goal 1 now; Goal 2 can be added in a next change.
