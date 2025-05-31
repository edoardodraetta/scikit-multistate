"""
Base classes for survival analysis models
==========================================

This module provides abstract base classes that define common interfaces
for all survival analysis models in scikit-multistate.

Classes
-------
- BaseMultiStateEstimator : Abstract base for all multistate models
- CompetingRisksModel : Foundation for competing risks analysis
"""

from .estimator import BaseMultiStateEstimator
from .competing_risks import CompetingRisksModel

__all__ = [
    'BaseMultiStateEstimator',
    'CompetingRisksModel',
]