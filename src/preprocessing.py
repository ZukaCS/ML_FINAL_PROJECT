"""Cleaning stage of the pipeline: raw competition frame in, one merged and
imputed frame out. No features are built here — that lives in feateng.py.

Assignment constraint: the saved pipeline must run on the raw, unpreprocessed
test.csv. Cleaner therefore carries the two side tables (stores.csv,
features.csv) inside the object, so transform() needs nothing but the raw
train/test frame.
"""

from __future__ import annotations

import pandas as pd

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]


class Cleaner:
    """Merge side tables and impute their gaps.

    Parameters
    ----------
    stores, features : raw stores.csv / features.csv frames.
    clip_negatives : clip negative Weekly_Sales (product returns, 0.3% of
        rows) to 0. Kept as a knob so each model's {X}_Preprocessing MLflow
        run can A/B it instead of assuming.
    """

    def __init__(self, stores: pd.DataFrame, features: pd.DataFrame,
                 clip_negatives: bool = True):
        self.clip_negatives = clip_negatives
        self.stores = stores.copy()

        side = features.copy()
        side["Date"] = pd.to_datetime(side["Date"])
        side = side.sort_values(["Store", "Date"])
        # NaN markdown = no promotion recorded -> 0, not a mean fill
        side[MARKDOWN_COLS] = side[MARKDOWN_COLS].fillna(0.0)
        # CPI/Unemployment are NaN only in the tail of the test period;
        # per-store forward fill uses past values only, so no leakage
        side[["CPI", "Unemployment"]] = (
            side.groupby("Store")[["CPI", "Unemployment"]].ffill()
        )
        # train/test carry their own IsHoliday; drop the duplicate column
        self.features = side.drop(columns=["IsHoliday"])

    def fit(self, df: pd.DataFrame, y=None) -> "Cleaner":
        return self  # all state comes from the side tables at __init__

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.merge(self.stores, on="Store", how="left")
        out = out.merge(self.features, on=["Store", "Date"], how="left")
        out["IsHoliday"] = out["IsHoliday"].astype(int)
        if self.clip_negatives and "Weekly_Sales" in out.columns:
            out["Weekly_Sales"] = out["Weekly_Sales"].clip(lower=0)
        return out

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_params(self) -> dict:
        return {"clip_negatives": self.clip_negatives,
                "markdown_impute": "zero",
                "cpi_unemp_impute": "ffill_per_store"}
