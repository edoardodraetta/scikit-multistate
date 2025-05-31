"""
Advanced preprocessing for survival analysis
===========================================

Scikit-learn compatible transformers and utilities for preparing
survival analysis data.

Classes
-------
- SurvivalTimeScaler : Scale survival times preserving censoring
- CompetingRiskEncoder : Encode competing risk events
- TimeVaryingTransformer : Handle time-varying covariates
"""

from .transforms import (
    SurvivalTimeScaler, CompetingRiskEncoder, 
    TimeVaryingTransformer
)

__all__ = [
    'SurvivalTimeScaler',
    'CompetingRiskEncoder', 
    'TimeVaryingTransformer',
]