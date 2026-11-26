"""From the Delta training set to model-ready pandas frames (Phase 5).

The training set is small (~53k rows x ~40 columns), so models are trained single-node with
scikit-learn / LightGBM on pandas, while the data work before it stays in Spark.

`prepare_features` is THE function that turns rows into model input: training uses it here
and batch scoring (Phase 6) must use it too, so a model never sees columns in a different
order or type than it was trained with. All features become float64 (NULL -> NaN): one
uniform type in the MLflow signature, and NaN is what scikit-learn imputers and LightGBM's
missing-value handling expect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from modeling.build_training_set import KEY_COLUMNS, LABEL_COLUMN, MODEL_FEATURES, SPLITS


@dataclass
class Split:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    keys: pd.DataFrame  # customer_id, cutoff_date (same row order as X / y)

    @property
    def n(self) -> int:
        return len(self.y)


def prepare_features(df: pd.DataFrame, features: Sequence[str] = MODEL_FEATURES) -> pd.DataFrame:
    """Select `features` in this exact order, as float64 (missing -> NaN)."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return df.loc[:, list(features)].apply(pd.to_numeric, errors="raise").astype("float64")


def to_pandas(
    training_set, splits: Sequence[str] = SPLITS, features: Sequence[str] = MODEL_FEATURES
):
    """Spark training set -> pandas with only keys, split, features and label (in row order
    customer_id, cutoff_date, so results are reproducible)."""
    from pyspark.sql import functions as F

    cols = [*KEY_COLUMNS, "split", *features, LABEL_COLUMN]
    return (
        training_set.where(F.col("split").isin(list(splits)))
        .select(*cols)
        .orderBy(*KEY_COLUMNS)
        .toPandas()
    )


def split_frames(
    pdf: pd.DataFrame, features: Sequence[str] = MODEL_FEATURES, splits: Sequence[str] = SPLITS
) -> dict[str, Split]:
    """{split name: Split} for the splits present in `pdf`."""
    out = {}
    for name in splits:
        part = pdf[pdf["split"] == name].reset_index(drop=True)
        if part.empty:
            continue
        out[name] = Split(
            name=name,
            X=prepare_features(part, features),
            y=part[LABEL_COLUMN].astype(int).to_numpy(),
            keys=part.loc[:, list(KEY_COLUMNS)],
        )
    return out


def feature_health(X: pd.DataFrame) -> pd.DataFrame:
    """Per feature: NaN count, negatives, min, median, max (checked before log transforms)."""
    return pd.DataFrame(
        {
            "n_nan": X.isna().sum(),
            "n_negative": (X < 0).sum(),
            "min": X.min(),
            "median": X.median(),
            "max": X.max(),
        }
    )
