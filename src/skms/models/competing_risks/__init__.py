"""
Competing risks survival models
===============================

Models specifically designed for competing risks analysis, where subjects
can experience one of several mutually exclusive events.

Classes
-------
- CoxCompetingRisks : Cox proportional hazards for competing risks
- AFTCompetingRisks : Accelerated failure time for competing risks
"""

from .cox import CoxCompetingRisks
from .aft import AFTCompetingRisks

# Re-export the base class for convenience
from ..base.competing_risks import CompetingRisksModel

__all__ = [
    'CoxCompetingRisks',
    'AFTCompetingRisks',
    'CompetingRisksModel',  # For users who import from this submodule
]