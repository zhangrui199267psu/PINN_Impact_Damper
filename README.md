# PINN Impact Damper (Free-Free Multi-DOF Chain)

This repository contains a Physics-Informed Neural Network (PINN) workflow for a **multi-degree-of-freedom (multi-DOF) impact chain system** (current baseline: **20 DOFs**) with **free-free boundary conditions**.

The main study setup is:
- free-free chain dynamics,
- **left external mass initialized with velocity**,
- three baseline initial-velocity levels: **1, 2, and 10 m/s**,
- transient response prediction and post-processing,
- dispersion-curve generation and comparison (used for double-checking consistency of results).

Future/ongoing scaling studies are intended for higher-order systems, e.g. **50 DOFs** and **100 DOFs**.

---

## Repository Layout

```text
PINN_Impact_Damper/
├── PINN/
│   ├── pinn_impact_chain_solver.py
│   └── pinn_impact_chain_simulation.ipynb
├── Results/
│   ├── batch_summary.csv
│   ├── pinn_free_free_20dof_low.{npz,mat}
│   ├── pinn_free_free_20dof_medium.{npz,mat}
│   ├── pinn_free_free_20dof_high.{npz,mat}
│   └── 1
├── References/
│   ├── Article__On_the_dispersion_analysis_of_metaimpactor_lattices_using_parametric_Physics_Informed_Neural_Networks.pdf
│   └── Dispersion relation - impactors in linear chain
└── README.md
```

---

## Modeling Scope

### Physical system
- Multi-DOF lumped chain model (baseline dataset: 20 DOFs).
- Free-free ends.
- Left external mass has nonzero initial velocity excitation.

### Baseline velocity cases
- Low: **1 m/s**
- Medium: **2 m/s**
- High: **10 m/s**

### Outputs of interest
- Time-domain response of masses/DOFs.
- Spectral content for wave/dispersion interpretation.
- Dispersion curves for consistency checks between numerical/PINN-derived trends.

---

## Quick Start

### 1) Create and activate an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install numpy scipy matplotlib tensorflow jupyter
```

### 3) Run the notebook workflow

```bash
jupyter notebook PINN/pinn_impact_chain_simulation.ipynb
```

---

## Results and Validation Notes

- `Results/` includes low/medium/high free-free 20-DOF cases in `.npz` and `.mat` formats.
- `batch_summary.csv` aggregates case-level metrics.
- Dispersion-curve plots are used as a **double-check/validation step** for the predicted dynamic behavior.

---

## Next Experiments

Planned extensions include:
- increasing system size to **50 DOFs**,
- increasing system size to **100 DOFs**,
- checking how response and dispersion trends scale with DOF count.

---

## Notes

- PINN training and long-time simulation can be computationally intensive.
- TensorFlow performance depends on CPU/GPU/CUDA setup.
- `.mat` files are provided for MATLAB-compatible post-processing.
