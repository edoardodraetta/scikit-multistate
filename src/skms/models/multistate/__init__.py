"""
Multistate survival models
==========================

General multistate models that can handle transitions between multiple
non-absorbing states, not just terminal competing events.

Classes
-------
- MultiStateModel : General multistate survival model
- MultiStateSimulator : Simulation engine for fitted models
"""

from .core import MultiStateModel, MultiStateSimulator

# Re-export base class for convenience
from ..base.estimator import BaseMultiStateEstimator

__all__ = [
    'MultiStateModel',
    'MultiStateSimulator',
    'BaseMultiStateEstimator',
]