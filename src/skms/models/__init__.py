"""
Survival analysis models
========================

This module contains all statistical models for survival analysis,
organized by methodology and use case.

Submodules
----------
- base : Abstract base classes and common functionality
- competing_risks : Models for competing risks analysis
- multistate : General multistate survival models
"""

# Import the most commonly used base class
from .base.competing_risks import CompetingRisksModel
from .base.estimator import BaseMultiStateEstimator

# Import main model implementations
from .competing_risks.cox import CoxCompetingRisks
from .competing_risks.aft import AFTCompetingRisks
from .multistate.core import MultiStateModel

__all__ = [
    # Base classes for advanced users
    "CompetingRisksModel",
    "BaseMultiStateEstimator",
    # Concrete implementations
    "CoxCompetingRisks",
    "AFTCompetingRisks",
    "MultiStateModel",
]
