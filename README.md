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


## New notebook: plot multiple dispersion curves together

A new notebook is provided for comparing multiple simulation outputs in one figure:

- `Dispersion/plot_dispersion_together.ipynb`

Use it to load `Result/pinn_data.mat` and any new file (e.g. `Result/new_data.mat`) and automatically detect passbands from DOS, extract top 3 bands, and plot all Band 1/2/3 curves from all MAT files together in one combined plot. If a file has fewer detected passbands, set `force_three_bands = True` in the notebook settings to force exactly 3 plotted bands.


## Parametric PINN folder (force-only sweep)

A new folder `parametric_PINN/` is added for force-parametric runs with **constant structural parameters**:

- `parametric_PINN/parametric_pinn_force_sweep.py`
  - Script to run multiple cases where only `phi1` and `phi2` vary.
- `parametric_PINN/parametric_pinn_force_sweep.ipynb`
  - Notebook interface for the same workflow.

Outputs are saved to `Result/` as:

- `pinn_data_phi1_<...>_phi2_<...>.mat`


## Continuity-based multi-ridge tracker

A new file is added for branch-continuity tracking in `(k, ω)` space:

- `Dispersion/continuity_multi_ridge_tracker.py`

This method tracks multiple ridges with a dynamic-programming continuity objective (jump penalty), which is more robust than purely DOS-window-based band splitting when branches are noisy or close.


- `Dispersion/plot_dispersion_together_continuity.ipynb`
  - Calls `continuity_multi_ridge_tracker.py` and overlays continuity-tracked 3 ridges from all MAT files in one figure, with linear dispersion reference.


## Bloch vs PINN comparison notebook

- `Dispersion/bloch_vs_pinn_comparison.ipynb`
  - Overlays analytical Bloch curves with PINN pointwise multi-band picking only (2 or 3 bands, no continuity tracker).


## Pointwise vs continuity comparison notebook

- `Dispersion/pointwise_vs_continuity_comparison.ipynb`
  - Side-by-side comparison of pointwise ridge picking (`extract_ridge`) and continuity-based multi-ridge tracking on the same MAT files.


## True parametric PINN (single-model interpolation)

- `parametric_PINN/true_parametric_pinn.py`
  - True parametric architecture with inputs `(t, phi1, phi2)` for one-model in-range interpolation.
- `parametric_PINN/true_parametric_pinn.ipynb`
  - Example workflow: train on several `(phi1, phi2)` cases and predict an unseen in-range case.
