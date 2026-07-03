"""Preprocessing helpers for diagnostic CxG model pipelines.

These live in an importable module so sklearn Pipeline objects that reference
them can be serialised/deserialised by joblib across scripts, without
relying on ``__main__``.

Any function passed to ``FunctionTransformer`` (or a custom transformer class)
used inside the diagnostic pipeline must be defined here rather than inline in
a script.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _as_frame(X: Any) -> pd.DataFrame:
    """Return *X* as a :class:`pandas.DataFrame`."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _coerce_binary_frame(X: Any) -> pd.DataFrame:
    """Cast all columns of *X* to ``float``."""
    return _as_frame(X).astype(float)


class RareCategoryCollapser(BaseEstimator, TransformerMixin):
    """Collapse infrequent categories before one-hot encoding.

    Any category whose count in the training data is below *min_count* is
    replaced with *replacement*.  Unseen categories at transform time are also
    collapsed.
    """

    def __init__(self, min_count: int = 30, replacement: str = "__rare__") -> None:
        self.min_count = min_count
        self.replacement = replacement
        self.frequent_values_: dict[str, set[str]] = {}

    def fit(self, X: Any, y: Any = None) -> "RareCategoryCollapser":
        frame = _as_frame(X)
        self.frequent_values_ = {}
        for column in frame.columns:
            counts = frame[column].fillna("__missing__").astype(str).value_counts()
            self.frequent_values_[column] = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        frame = _as_frame(X).copy()
        for column in frame.columns:
            frequent = self.frequent_values_.get(column, set())
            values = frame[column].fillna("__missing__").astype(str)
            frame[column] = values.where(values.isin(frequent), self.replacement)
        return frame
