"""cXA Modeling Module.

This module contains the ultimate cXA model implementation:
- True xA model (isolates passer contribution)
- Credit distribution (position decay + contribution weighting)
- Opponent adjustment (defensive profile integration)
"""

from .ultimate_cxa_model import UltimateCxAModel

__all__ = ["UltimateCxAModel"]
