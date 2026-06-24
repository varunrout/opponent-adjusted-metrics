"""CxT Pipeline Module.

Provides data extraction and feature engineering pipelines for
Contextual Expected Threat (CxT) modeling.

Components:
- extract_progressions: Extract passes, carries, dribbles with xT values
- zone_mapping: Map pitch locations to macro-zones
"""

from __future__ import annotations

from .extract_progressions import build_progressions_dataset

__all__ = ["build_progressions_dataset"]
