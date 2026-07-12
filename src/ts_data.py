"""Time-series data preparation shared by every Darts model notebook.

Turns the cleaned long frame (output of preprocessing.Cleaner) into Darts
TimeSeries objects plus covariates, with three fixed decisions that are
logged in each model's {X}_Preprocessing run:

  1. every Store x Dept series is reindexed to a regular W-FRI grid and
     internal gap weeks are filled with 0 (missing week = nothing sold);
  2. every series is padded forward with 0 up to the common last training
     date, so predict(n) starts from the same week for all series;
  3. late-starting series keep their own start date (no backward padding).

Heavy build steps are cached per (frame checksum, options) so that repeated
pipeline fits inside one CV stage do not rebuild identical objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries

from src.feateng import CHRISTMASES, THANKSGIVINGS, _signed_days_to_nearest

_CACHE: dict = {}

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]
MAX_FOURIER_K = 8

# covariate column groups; every column is bounded to roughly [-1, 1]
CALENDAR_HOLIDAY_COLS = [
    "wk_sin", "wk_cos", "month_sin", "month_cos",
    "days_to_thanksgiving", "days_to_christmas",
    "is_thanksgiving_week", "is_pre_christmas_week", "is_holiday",
]
MACRO_COLS = [f"markdown{i}" for i in range(1, 6)] + [
    "temperature", "fuel_price", "cpi", "unemployment",
]
HOLIDAY_FLAG_COLS = ["is_holiday", "is_thanksgiving_week", "is_pre_christmas_week"]

PRESET_COLUMNS = {
    "none": [],
    "calendar_holiday": CALENDAR_HOLIDAY_COLS,
    "full": CALENDAR_HOLIDAY_COLS + MACRO_COLS,
}
for _k in (3, 5, 8):
    PRESET_COLUMNS[f"fourier{_k}_holiday"] = (
        [f"fourier_sin{i}" for i in range(1, _k + 1)]
        + [f"fourier_cos{i}" for i in range(1, _k + 1)]
        + HOLIDAY_FLAG_COLS
    )

# documentation table used by the {X}_Feature_Engineering runs
COVARIATE_DOC = pd.DataFrame([
    ("wk_sin/wk_cos", "Date", "sin/cos of 2*pi*dayofyear/365.25 (annual cycle)"),
    ("month_sin/month_cos", "Date", "sin/cos of 2*pi*(month-1)/12"),
    ("days_to_thanksgiving", "Date", "signed days to nearest Thanksgiving / 182"),
    ("days_to_christmas", "Date", "signed days to nearest Christmas / 182"),
    ("is_thanksgiving_week", "Date", "1 if week ends 0..6 days after Thanksgiving"),
    ("is_pre_christmas_week", "Date", "1 if week ends 0..7 days before Dec 25"),
    ("is_holiday", "features.IsHoliday", "bool cast to 0/1"),
    ("markdown1..5", "features.MarkDown1..5", "renamed, NaN filled with 0, minmax scaled"),
    ("temperature", "features.Temperature", "renamed, minmax scaled"),
    ("fuel_price", "features.Fuel_Price", "renamed, minmax scaled"),
    ("cpi", "features.CPI", "renamed, per-store ffill, minmax scaled"),
    ("unemployment", "features.Unemployment", "renamed, per-store ffill, minmax scaled"),
    ("fourier_sin1..8/fourier_cos1..8", "Date", "sin/cos of 2*pi*k*dayofyear/365.25"),
    ("static_store", "Store", "integer category code 0..44 (TFT embedding)"),
    ("static_dept", "Dept", "integer category code 0..98 (TFT embedding)"),
    ("static_type", "stores.Type", "ordinal: A=0.0, B=0.5, C=1.0"),
    ("static_size", "stores.Size", "minmax scaled over stores.csv"),
], columns=["column", "built_from", "transform"])

# fixed competition holiday weeks (week-ending Fridays), used by Prophet
PROPHET_HOLIDAYS = pd.DataFrame([
    {"holiday": name, "ds": pd.Timestamp(d), "lower_window": 0, "upper_window": 0}
    for name, dates in {
        "super_bowl":    ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
        "labor_day":     ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
        "thanksgiving":  ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
        "christmas":     ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
        "pre_christmas": ["2010-12-24", "2011-12-23", "2012-12-21", "2013-12-20"],
    }.items() for d in dates
])


def _frame_key(df: pd.DataFrame) -> tuple:
    """Cheap checksum of a frame: enough to distinguish CV fold slices."""
    total = float(df["Weekly_Sales"].sum()) if "Weekly_Sales" in df.columns else 0.0
    return (len(df), str(pd.to_datetime(df["Date"]).min())[:10],
            str(pd.to_datetime(df["Date"]).max())[:10], round(total, 2))


def weeks_ahead(train_end: pd.Timestamp, last_date: pd.Timestamp) -> int:
    """Forecast horizon in weeks from the training end to last_date."""
    days = (pd.Timestamp(last_date) - pd.Timestamp(train_end)).days
    if days <= 0 or days % 7 != 0:
        raise ValueError(f"prediction dates must extend past {train_end} "
                         f"in whole weeks, got {last_date}")
    return days // 7


def build_static_frame(clean_df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """One row of static covariates per Store x Dept pair.

    static_store / static_dept are integer category codes (id - 1) meant for
    TFT categorical embeddings: nominal identities carry no order, so they
    must not enter the network as scaled numbers. The codes are id-based
    (not fit-dependent), so a pair always maps to the same embedding row.
    static_type is ordinal (A > B > C by size) and static_size continuous.
    """
    pairs = clean_df[["Store", "Dept", "Type", "Size"]].drop_duplicates(["Store", "Dept"])
    smin, smax = float(stores["Size"].min()), float(stores["Size"].max())
    out = pd.DataFrame({
        "static_store": pairs["Store"].astype(float) - 1.0,   # codes 0..44
        "static_dept": pairs["Dept"].astype(float) - 1.0,     # codes 0..98
        "static_type": pairs["Type"].map({"A": 0.0, "B": 0.5, "C": 1.0}).astype(float),
        "static_size": (pairs["Size"].astype(float) - smin) / (smax - smin),
    })
    out.index = pd.MultiIndex.from_arrays([pairs["Store"], pairs["Dept"]])
    return out


def build_target_series(clean_df: pd.DataFrame, log_target: bool = False,
                        scale: bool = False, statics: pd.DataFrame | None = None,
                        dtype=np.float32, cache: bool = True):
    """Cleaned long frame -> ({(Store, Dept): TimeSeries}, scale params, report).

    scale=True applies per-series minmax AFTER the optional log1p, so the
    inverse at predict time is: unscale first, then expm1.
    """
    key = ("target", _frame_key(clean_df), bool(log_target), bool(scale),
           statics is not None, np.dtype(dtype).name)
    if cache and key in _CACHE:
        return _CACHE[key]

    wide = clean_df.pivot(index="Date", columns=["Store", "Dept"],
                          values="Weekly_Sales")
    full_idx = pd.date_range(clean_df["Date"].min(), clean_df["Date"].max(),
                             freq="W-FRI")
    wide = wide.reindex(full_idx)
    arr = wide.to_numpy(dtype=np.float64)

    series, params = {}, {}
    n_gap = n_tail_series = n_tail_weeks = n_late = 0
    for j, pair in enumerate(wide.columns):
        col = arr[:, j]
        notna = ~np.isnan(col)
        first = int(notna.argmax())
        last = len(col) - 1 - int(notna[::-1].argmax())
        n_gap += int(np.isnan(col[first:last + 1]).sum())
        n_tail_series += int(last < len(col) - 1)
        n_tail_weeks += len(col) - 1 - last
        n_late += int(first > 0)

        vals = np.nan_to_num(col[first:], nan=0.0)
        if log_target:
            vals = np.log1p(np.clip(vals, 0.0, None))
        vmin, vmax = float(vals.min()), float(vals.max())
        if scale:
            vals = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
        sc = statics.loc[[pair]].reset_index(drop=True) if statics is not None else None
        series[pair] = TimeSeries.from_times_and_values(
            full_idx[first:], vals.astype(dtype).reshape(-1, 1),
            columns=["Weekly_Sales"], static_covariates=sc)
        params[pair] = (vmin, vmax)

    report = {
        "n_series": len(series),
        "n_weeks_common_grid": len(full_idx),
        "n_gap_weeks_filled": n_gap,
        "n_tail_padded_series": n_tail_series,
        "n_tail_padded_weeks": n_tail_weeks,
        "n_late_start_series": n_late,
    }
    result = (series, params, report)
    if cache:
        _CACHE[key] = result
    return result


def build_weight_series(features: pd.DataFrame, holiday_weight: float = 5.0,
                        cache: bool = True) -> TimeSeries:
    """Per-time-step sample weights for torch-model training: holiday_weight
    on IsHoliday weeks, 1 elsewhere. With an L1 loss this makes the training
    objective exactly the competition WMAE (which counts holiday weeks 5x),
    mirroring the sample_weight=5 the tree models trained with.

    One global series covers 2010-02 through 2013-07; Darts slices it to each
    target's span internally.
    """
    key = ("weights", float(holiday_weight), len(features))
    if cache and key in _CACHE:
        return _CACHE[key]
    f = features[["Date", "IsHoliday"]].copy()
    f["Date"] = pd.to_datetime(f["Date"])
    if (f.groupby("Date")["IsHoliday"].nunique() != 1).any():
        raise ValueError("IsHoliday differs across stores for the same week")
    cal = f.drop_duplicates("Date").sort_values("Date")
    w = np.where(cal["IsHoliday"], float(holiday_weight), 1.0).astype(np.float32)
    ts = TimeSeries.from_times_and_values(
        pd.DatetimeIndex(cal["Date"]), w.reshape(-1, 1), columns=["weight"])
    if cache:
        _CACHE[key] = ts
    return ts


def _covariate_base(features: pd.DataFrame) -> dict:
    """All covariate columns per store, over the full features.csv range
    (2010-02 .. 2013-07, which covers the whole test horizon).

    Scaling constants come from the full frame: covariates are exogenous
    inputs supplied by the competition for the entire period, so no target
    information leaks through them.
    """
    key = ("cov_base", len(features))
    if key in _CACHE:
        return _CACHE[key]

    side = features.copy()
    side["Date"] = pd.to_datetime(side["Date"])
    side = side.sort_values(["Store", "Date"])
    side[MARKDOWN_COLS] = side[MARKDOWN_COLS].fillna(0.0)
    side[["CPI", "Unemployment"]] = (
        side.groupby("Store")[["CPI", "Unemployment"]].ffill())

    d = side["Date"]
    doy = d.dt.dayofyear.to_numpy(dtype=np.float64)
    for k in range(1, MAX_FOURIER_K + 1):
        side[f"fourier_sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        side[f"fourier_cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    side["wk_sin"] = side["fourier_sin1"]
    side["wk_cos"] = side["fourier_cos1"]
    month = d.dt.month.to_numpy(dtype=np.float64)
    side["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    side["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    dist_t = _signed_days_to_nearest(d, THANKSGIVINGS)
    dist_c = _signed_days_to_nearest(d, CHRISTMASES)
    side["is_thanksgiving_week"] = ((dist_t >= -6) & (dist_t <= 0)).astype(float)
    side["is_pre_christmas_week"] = ((dist_c >= 0) & (dist_c <= 7)).astype(float)
    side["days_to_thanksgiving"] = dist_t / 182.0
    side["days_to_christmas"] = dist_c / 182.0
    side["is_holiday"] = side["IsHoliday"].astype(float)

    renames = {"Temperature": "temperature", "Fuel_Price": "fuel_price",
               "CPI": "cpi", "Unemployment": "unemployment",
               **{c: c.lower() for c in MARKDOWN_COLS}}
    side = side.rename(columns=renames)
    for c in MACRO_COLS:
        cmin, cmax = float(side[c].min()), float(side[c].max())
        side[c] = (side[c] - cmin) / (cmax - cmin) if cmax > cmin else 0.0

    all_cols = sorted(set(c for cols in PRESET_COLUMNS.values() for c in cols))
    if side[all_cols].isna().any().any():
        raise ValueError("NaN left in covariates after imputation")
    base = {int(s): g.set_index("Date")[all_cols]
            for s, g in side.groupby("Store")}
    _CACHE[key] = base
    return base


def build_covariates(features: pd.DataFrame, preset: str,
                     dtype=np.float32, cache: bool = True):
    """{Store: future-covariate TimeSeries} for one preset (None for "none")."""
    if preset == "none":
        return None
    if preset not in PRESET_COLUMNS:
        raise KeyError(f"unknown covariate preset {preset!r}")
    key = ("covs", preset, len(features), np.dtype(dtype).name)
    if cache and key in _CACHE:
        return _CACHE[key]

    cols = PRESET_COLUMNS[preset]
    base = _covariate_base(features)
    out = {store: TimeSeries.from_times_and_values(
               frame.index, frame[cols].to_numpy(dtype=dtype), columns=cols)
           for store, frame in base.items()}
    if cache:
        _CACHE[key] = out
    return out


def covariate_table(preset: str, static_cols=()) -> pd.DataFrame:
    """Documentation rows (column / built_from / transform) for one preset."""
    wanted = list(PRESET_COLUMNS[preset]) + list(static_cols)

    def matches(doc_name: str) -> bool:
        heads = [doc_name.split("..")[0].split("/")[0].rstrip("0123456789")]
        if "/" in doc_name:
            heads.append(doc_name.split("/")[1].split("..")[0].rstrip("0123456789"))
        return any(w.rstrip("0123456789").startswith(h) for w in wanted for h in heads)

    return COVARIATE_DOC[COVARIATE_DOC["column"].map(matches)].reset_index(drop=True)


def stratified_sample_pairs(train: pd.DataFrame, stores: pd.DataFrame,
                            n: int = 300, seed: int = 42,
                            min_weeks: int = 120,
                            latest_start: str = "2010-04-02"):
    """Sample of Store x Dept pairs for local-model CV, stratified by store
    Type and volume quintile. Restricted to long, early-starting series so
    every CV fold has enough history to fit on.

    Returns (list of (Store, Dept) tuples, composition frame for logging).
    """
    g = train.copy()
    g["Date"] = pd.to_datetime(g["Date"])
    stats = (g.groupby(["Store", "Dept"])
             .agg(n_weeks=("Date", "count"), first_date=("Date", "min"),
                  mean_sales=("Weekly_Sales", "mean"))
             .reset_index()
             .merge(stores[["Store", "Type"]], on="Store"))
    eligible = stats[(stats["n_weeks"] >= min_weeks)
                     & (stats["first_date"] <= pd.Timestamp(latest_start))].copy()
    eligible["vol_q"] = pd.qcut(eligible["mean_sales"], 5,
                                labels=[f"q{i}" for i in range(1, 6)])

    frac = min(1.0, n / len(eligible))
    samp = (eligible.groupby(["Type", "vol_q"], observed=True)
            .sample(frac=frac, random_state=seed))
    rng = np.random.default_rng(seed)
    if len(samp) > n:
        samp = samp.sample(n, random_state=seed)
    elif len(samp) < n:
        extra = eligible.drop(samp.index)
        take = extra.iloc[rng.choice(len(extra), size=min(n - len(samp), len(extra)),
                                     replace=False)]
        samp = pd.concat([samp, take])

    pairs = list(map(tuple, samp[["Store", "Dept"]].to_numpy()))
    composition = (samp.groupby(["Type", "vol_q"], observed=True)
                   .size().rename("n_pairs").reset_index())
    return pairs, composition
