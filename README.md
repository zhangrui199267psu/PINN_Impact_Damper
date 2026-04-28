# PINN_Impact_Damper

Physics-informed / physics-guided workflows for impact-damper lattice analysis, including:

- dispersion-theory utilities,
- PINN post-processing and PINN-vs-Bloch comparison,
- free-free 20-DOF simulation workflows,
- parametric velocity-conditioned surrogate experiments.

---

## Repository Structure

```text
PINN_Impact_Damper/
├── Dispersion/
│   ├── dispersion_theory.py
│   ├── dispersion_tutorial.ipynb
│   └── plot_dispersion_together_3A.ipynb
├── PINN/
│   ├── pinn_ndof_chain_tf2.py
│   ├── pinn_ndof_chain_sim_tf2_freq2_A10.ipynb
│   └── pinn_dispersion_from_mat.py
├── PINN_free_free/
│   ├── README.md
│   ├── pinn_ndof_chain_tf2_free_free_no_force.py
│   ├── pinn_ndof_chain_sim_tf2_free_free_icv.ipynb
│   └── pinn_ndof_chain_parametric_tf2.py
├── Predicted_Data/
├── References/
└── README.md
```

---

## What Each Part Does

### 1) `Dispersion/`

- `dispersion_theory.py`  
  Analytical helpers for monoatomic / mass-in-mass / impact-damper style branch calculations and plotting.

- `dispersion_tutorial.ipynb`  
  Tutorial-style notebook for dispersion theory, overlays, and effective-parameter interpretation.

- `plot_dispersion_together_3A.ipynb`  
  Multi-case analysis notebook (MAT loading, spectral estimation, passband extraction, PINN-vs-Bloch overlays, impulse-guided parameter cues).

### 2) `PINN/`

- `pinn_ndof_chain_tf2.py`  
  Base TF2 PINN implementation for chain dynamics.

- `pinn_ndof_chain_sim_tf2_freq2_A10.ipynb`  
  Canonical simulation notebook corresponding to original forcing setup.

- `pinn_dispersion_from_mat.py`  
  Utilities for loading/processing predicted results, ridge/heatmap extraction, comparison plotting, and impulse estimation.

### 3) `PINN_free_free/`

- Free-free/no-force workflows (20 DOFs) with initial-velocity excitation.
- Includes both notebook workflow and standalone parametric velocity-conditioned surrogate script.

See `PINN_free_free/README.md` for details.

---

## Quick Start

## A) Dispersion workflows

```bash
jupyter notebook Dispersion/dispersion_tutorial.ipynb
```

```bash
jupyter notebook Dispersion/plot_dispersion_together_3A.ipynb
```

## B) Original PINN workflow

```bash
jupyter notebook PINN/pinn_ndof_chain_sim_tf2_freq2_A10.ipynb
```

## C) Free-free workflow

```bash
jupyter notebook PINN_free_free/pinn_ndof_chain_sim_tf2_free_free_icv.ipynb
```

## D) Parametric velocity-conditioned surrogate (script)

```bash
python PINN_free_free/pinn_ndof_chain_parametric_tf2.py
```

---

## Data and References

- `Predicted_Data/` contains example MAT files and related generated assets.
- `References/` contains background documents used to guide interpretation and comparisons.

---

## Typical Outputs

Depending on notebook/script:

- dispersion plots and overlays,
- PINN spectral heatmaps and ridge diagnostics,
- impact-time / impulse statistics,
- saved per-case results (`.npz`, optional `.mat`),
- parametric model artifacts (`.keras`, metadata `.npz`).

---

## Notes

- Some notebooks/scripts are computationally heavy (PINN training + long simulations).
- For parametric surrogates, best performance requires training velocities spanning your intended inference range.
- Free-free notebook currently uses a target simulation time criterion and can process multiple velocity cases in one run.
