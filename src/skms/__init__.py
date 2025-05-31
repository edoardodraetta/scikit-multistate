"""
scikit-multistate: Multistate survival analysis for Python
===========================================================

A Python package for multistate survival analysis and competing risks modeling,
built on top of scikit-learn.

Main Classes
------------
- CoxCompetingRisks : Cox proportional hazards for competing risks
- AFTCompetingRisks : Accelerated failure time for competing risks
- MultiStateModel : General multistate survival model
- CompetingRisksModel : Base competing risks framework

Quick Start
-----------
>>> from skms import CoxCompetingRisks, load_aidssi
>>> data = load_aidssi()
>>> model = CoxCompetingRisks()
>>> model.fit(data, duration_col='time', event_col='event')
"""

# Core user-facing models (most common imports)
# from .models.competing_risks.cox import CoxCompetingRisks
# from .models.competing_risks.aft import AFTCompetingRisks
# from .models.multistate.core import MultiStateModel
# from .models.base.competing_risks import CompetingRisksModel

# Essential data utilities
from .data import (
    load_aidssi,
    list_available_datasets,
)

# Key preprocessing tools
# from .preprocessing.transforms import SurvivalTimeScaler, CompetingRiskEncoder

# # Main visualization functions
from .visualization.state_diagram import StateDiagramGenerator

# Version info
# from ._version import __version__

# What gets imported with "from skms import *"
__all__ = [
    # Models - prioritized by common usage
    # Data loading
    "load_aidssi",
    "list_available_datasets",
    # Visualization
    "StateDiagramGenerator",
    # Meta
    # "__version__",
]
