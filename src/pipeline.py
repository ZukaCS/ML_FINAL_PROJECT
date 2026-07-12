"""The end-to-end pipeline object the assignment asks us to save:
raw competition frame in -> predictions out (cleaning + features + model).
"""

from __future__ import annotations

import numpy as np


class LogTarget:
    """log1p target transform. A named class (not a lambda) so the fitted
    pipeline stays picklable for MLflow."""

    name = "log1p"

    def forward(self, y):
        return np.log1p(np.clip(y, 0, None))

    def inverse(self, p):
        return np.expm1(p)


class SeasonalNaive:
    """Baseline model: predict same week last year (the lag_52 feature).
    Use with FeatureBuilder(fill_lags=True) so cold-start rows get the
    department-median fallback instead of NaN."""

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return X["lag_52"].to_numpy(dtype=float)


class WalmartPipeline:
    """Cleaner -> FeatureBuilder -> model, as one fit/predict object.

    holiday_weight : sample weight for IsHoliday training rows. WMAE counts
        holiday weeks 5x, so holiday_weight=5 makes the training loss match
        the metric; 1.0 disables weighting.
    """

    def __init__(self, cleaner, feature_builder, model, target_transform=None,
                 holiday_weight: float = 1.0):
        self.cleaner = cleaner
        self.feature_builder = feature_builder
        self.model = model
        self.target_transform = target_transform
        self.holiday_weight = holiday_weight

    def fit(self, raw_train):
        df = self.cleaner.fit(raw_train).transform(raw_train)
        X = self.feature_builder.fit(df).transform(df)
        y = df["Weekly_Sales"]
        if self.target_transform is not None:
            y = self.target_transform.forward(y)
        if self.holiday_weight != 1.0:
            w = np.where(df["IsHoliday"] == 1, self.holiday_weight, 1.0)
            self.model.fit(X, y, sample_weight=w)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, raw_df):
        df = self.cleaner.transform(raw_df)
        X = self.feature_builder.transform(df)
        pred = np.asarray(self.model.predict(X), dtype=float)
        if self.target_transform is not None:
            pred = self.target_transform.inverse(pred)
        return pred

    def get_params(self) -> dict:
        return {**self.cleaner.get_params(),
                **self.feature_builder.get_params(),
                "target_transform": getattr(self.target_transform, "name", "none"),
                "holiday_weight": self.holiday_weight,
                "model_class": type(self.model).__name__}
