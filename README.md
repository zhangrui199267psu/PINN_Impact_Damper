# PINN Impact Damper

This repository contains code and results for studying wave propagation and transient response in an impact-damper chain using a Physics-Informed Neural Network (PINN) workflow.

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

## What is Included

- **PINN solver script** (`PINN/pinn_impact_chain_solver.py`) for model setup/training/inference.
- **Notebook workflow** (`PINN/pinn_impact_chain_simulation.ipynb`) for interactive experiments and visualization.
- **Precomputed result sets** in `.npz` and `.mat` formats under `Results/`.
- **Reference materials** under `References/` for theoretical context.

## Quick Start

### 1) Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

Install dependencies required by the solver/notebook (for example: `numpy`, `scipy`, `matplotlib`, `tensorflow`, `jupyter`).

```bash
pip install numpy scipy matplotlib tensorflow jupyter
```

> If your local setup already has a project-specific requirements file, prefer that over the generic command above.

### 3) Run the notebook

```bash
jupyter notebook PINN/pinn_impact_chain_simulation.ipynb
```

### 4) Run the solver script

```bash
python PINN/pinn_impact_chain_solver.py
```

## Results

The `Results/` directory contains low/medium/high case outputs and a summary CSV (`batch_summary.csv`) that can be used for post-processing and comparison plots.

## Notes

- Training and simulation can be computationally expensive.
- TensorFlow/GPU behavior depends on your local CUDA/cuDNN setup.
- `.mat` files are provided for MATLAB-compatible analysis.
