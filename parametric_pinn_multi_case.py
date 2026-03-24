"""
parametric_pinn_multi_case.py — compatibility shim
---------------------------------------------------
All content from this module has been merged into parametric_impact_pinn_tf1.py.
This file re-exports everything so that existing imports continue to work.
"""

from parametric_impact_pinn_tf1 import (  # noqa: F401
    lhs_sample,
    find_impact_time,
    _impact_update,
    ParametricSequentialPredictor,
    train_parametric_pinn,
    ParametricImpactPINN,
    SequentialParametricImpactPINN,
)
