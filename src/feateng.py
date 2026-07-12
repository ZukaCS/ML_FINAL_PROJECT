"""Feature engineering stage: cleaned frame in, model-ready X out.

Every history feature is a *lookup* into the training sales memorized by
fit() — the same code path serves train rows, CV validation rows and the
real test set. That is what keeps the pipeline leakage-free and able to run
on the raw test.csv (assignment requirement).

Only the annual-seasonality lags are used: lag-52 (same week last year,
r = 0.70 in the EDA) and lag-104 (same week two years ago, r = 0.54), plus a
small smoothing window around lag-52. The 39-week test horizon makes shorter
lags unobservable at prediction time, and their autocorrelation is ~0 anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEK = pd.Timedelta(weeks=1)

# Fixed by the competition (2010-2013); boundary years included so that
# "nearest holiday" is well-defined at the edges of the data.
THANKSGIVINGS = pd.to_datetime([
    "2009-11-26", "2010-11-25", "2011-11-24",
    "2012-11-22", "2013-11-28", "2014-11-27",
])
CHRISTMASES = pd.to_datetime([f"{y}-12-25" for y in range(2009, 2015)])

FEATURE_GROUPS = {
    "lags":        ["lag_52", "lag_104"],
    "lag_windows": ["roll_mean_52", "roll_std_52"],
    "calendar":    ["Year", "Month", "Week"],
    "holiday":     ["days_to_thanksgiving", "days_to_christmas",
                    "is_thanksgiving_week", "is_pre_christmas_week", "IsHoliday"],
    "statics":     ["Size", "Type"],
    "markdowns":   [f"MarkDown{i}" for i in range(1, 6)],
    "macro":       ["Temperature", "Fuel_Price", "CPI", "Unemployment"],
}
ID_COLS = ["Store", "Dept"]  # identify the series; always included


def _signed_days_to_nearest(dates: pd.Series, holidays: pd.DatetimeIndex) -> np.ndarray:
    """Signed distance in days to the nearest holiday (positive = upcoming).

    Handles the two EDA traps at once: the Christmas spike drifting between
    ISO weeks 51/52, and IsHoliday flagging the (below-average) week that
    contains Dec 25 instead of the peak week before it.
    """
    d = dates.values.astype("datetime64[D]").astype(int)
    h = np.sort(holidays.values.astype("datetime64[D]").astype(int))
    pos = np.searchsorted(h, d)
    prev = h[np.clip(pos - 1, 0, len(h) - 1)]
    nxt = h[np.clip(pos, 0, len(h) - 1)]
    to_next, since_prev = nxt - d, d - prev
    return np.where(to_next <= since_prev, to_next, -since_prev)


class FeatureBuilder:
    """fit() memorizes training history; transform() builds features by lookup.

    Parameters
    ----------
    groups : which FEATURE_GROUPS to emit. Store/Dept always included.
    drop_cols : individual columns removed after group selection
        (used by the {X}_Feature_Selection stage).
    fill_lags : replace missing lag values (cold-start pairs, first year of
        train) with the department median memorized in fit(). LightGBM
        handles NaN natively so trees keep this off; the seasonal-naive
        baseline and NN pipelines turn it on.
    """

    def __init__(self,
                 groups=("lags", "lag_windows", "calendar", "holiday", "statics"),
                 drop_cols=(), fill_lags: bool = False):
        self.groups = list(groups)
        self.drop_cols = list(drop_cols)
        self.fill_lags = fill_lags

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, y=None) -> "FeatureBuilder":
        s = df.set_index(["Store", "Dept", "Date"])["Weekly_Sales"]
        self.sales_ = s[~s.index.duplicated()].sort_index()
        # cold-start fallback: dept median across stores, then global median
        self.dept_median_ = df.groupby("Dept")["Weekly_Sales"].median()
        self.global_median_ = float(df["Weekly_Sales"].median())
        # freeze category sets so train/test encodings can never diverge
        self.cats_ = {c: pd.CategoricalDtype(sorted(df[c].unique()))
                      for c in ("Store", "Dept", "Type") if c in df.columns}
        return self

    # ------------------------------------------------------------ internals
    def _lag(self, df: pd.DataFrame, weeks: int) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays(
            [df["Store"], df["Dept"], df["Date"] - weeks * WEEK])
        return self.sales_.reindex(idx).to_numpy()

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- history lookups (same mechanism for train, CV-val and test)
        need = set()
        if "lags" in self.groups:
            need |= {52, 104}
        if "lag_windows" in self.groups:
            need |= {50, 51, 52, 53, 54}
        cache = {k: self._lag(out, k) for k in sorted(need)}

        if "lags" in self.groups:
            out["lag_52"], out["lag_104"] = cache[52], cache[104]
        if "lag_windows" in self.groups:
            win52 = pd.DataFrame({k: cache[k] for k in (50, 51, 52, 53, 54)})
            out["roll_mean_52"] = win52.mean(axis=1).to_numpy()
            out["roll_std_52"] = win52.std(axis=1).to_numpy()

        if self.fill_lags:
            fallback = (out["Dept"].map(self.dept_median_)
                        .fillna(self.global_median_))
            for c in FEATURE_GROUPS["lags"] + FEATURE_GROUPS["lag_windows"]:
                if c in out.columns:
                    out[c] = out[c].fillna(fallback)

        # --- pure functions of Date
        if "calendar" in self.groups:
            out["Year"] = out["Date"].dt.year
            out["Month"] = out["Date"].dt.month
            out["Week"] = out["Date"].dt.isocalendar().week.astype(int)
        if "holiday" in self.groups:
            out["days_to_thanksgiving"] = _signed_days_to_nearest(out["Date"], THANKSGIVINGS)
            out["days_to_christmas"] = _signed_days_to_nearest(out["Date"], CHRISTMASES)
            # week ending the Friday right after Thanksgiving Thursday
            out["is_thanksgiving_week"] = out["days_to_thanksgiving"].between(-6, 0).astype(int)
            # the true sales peak: the week *before* Dec 25 (IsHoliday misses it)
            out["is_pre_christmas_week"] = out["days_to_christmas"].between(0, 7).astype(int)

        # --- column selection + frozen categorical encodings
        cols = ID_COLS + [c for g in self.groups for c in FEATURE_GROUPS[g]]
        cols = [c for c in cols if c not in self.drop_cols]
        X = out[cols].copy()
        for c, dtype in self.cats_.items():
            if c in X.columns:
                X[c] = X[c].astype(dtype)
        return X

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_params(self) -> dict:
        return {"feature_groups": self.groups,
                "drop_cols": self.drop_cols,
                "fill_lags": self.fill_lags}
