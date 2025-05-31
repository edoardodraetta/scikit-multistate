"""
Visualization tools for survival analysis
==========================================

Functions and classes for creating publication-ready plots of
survival analysis results.

Functions
---------
- plot_state_diagram : Create state transition diagrams
- plot_transition_probabilities : Plot transition probabilities over time
- plot_cumulative_incidence : Plot cumulative incidence functions
- plot_survival_curves : Plot survival curves with confidence intervals

Classes
-------
- StateDiagramGenerator : Advanced state diagram creation
"""

from .state_diagram import StateDiagramGenerator

__all__ = [
    # Classes
    "StateDiagramGenerator",
]
