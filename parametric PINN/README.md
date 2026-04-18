# Parametric PINN (single model for full 50-impact horizon)

This folder contains a **true parametric PINN** that learns one mapping:

\[(t, \phi_1, \phi_2) \rightarrow x(t;\phi_1,\phi_2)\]

with
- \(\phi_1 \in [1, 2]\)
- \(\phi_2 \in [10, 20]\)

using **one neural network** for the **entire response window** (not per-impact segment).

## Files
- `parametric_pinn_50_impacts.py`
  - Samples parameter pairs (default 20 samples).
  - Trains one PINN over all sampled parameters and full time range.
  - Predicts response for unseen in-range parameters.
  - Extracts approximate 50 impact times from a learned contact indicator.
- `parametric_pinn_50_impacts.ipynb`
  - Notebook wrapper to run training/prediction interactively.

## Quick start
```bash
python "parametric PINN/parametric_pinn_50_impacts.py"
```

Or open and run:

- `parametric PINN/parametric_pinn_50_impacts.ipynb`

## Notes
- The implementation uses a smooth contact-force surrogate so the PINN can be trained end-to-end over long horizons with many impacts.
- If you need the exact original impact law (with explicit impactor state updates), this file is a strong starting point and can be extended to multi-output state PINNs.
