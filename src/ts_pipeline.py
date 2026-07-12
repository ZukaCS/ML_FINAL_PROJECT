"""Darts model wrappers with the same fit(raw_df) / predict(raw_df) interface
as pipeline.WalmartPipeline, so validation.evaluate, experiment_utils and the
MLflow logging conventions work unchanged for every architecture.

GlobalDartsPipeline : one torch model trained jointly on all series
                      (N-BEATS, DLinear, TFT).
LocalDartsPipeline  : one statistical model fitted per series, in parallel
                      (ARIMA, SARIMA, Prophet).

Both fall back to the seasonal-naive pipeline (same-week-last-year with a
department-median cold-start fill) for any row the model cannot predict:
pairs never seen in training and series shorter than the model minimum.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import ts_data
from src.feateng import FeatureBuilder
from src.pipeline import SeasonalNaive, WalmartPipeline
from src.preprocessing import Cleaner

_FALLBACK_CACHE: dict = {}


def _build_fallback(raw_train, stores, features):
    key = ts_data._frame_key(raw_train)
    if key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[key] = WalmartPipeline(
            Cleaner(stores, features),
            FeatureBuilder(groups=["lags"], fill_lags=True),
            SeasonalNaive()).fit(raw_train)
    return _FALLBACK_CACHE[key]


class GlobalDartsPipeline:
    """Cleaner -> TimeSeries reshaping -> one global Darts torch model.

    Parameters
    ----------
    model_factory : zero-arg callable returning a FRESH Darts torch model
        (a new one is created on every fit so CV folds never share state).
    input_chunk_length / output_chunk_length : must match the model; used to
        decide which series are long enough to train on / predict from.
    covariate_preset : key of ts_data.PRESET_COLUMNS ("none" disables).
    log_target : apply log1p to Weekly_Sales before scaling.
    scale : per-series minmax scaling (fit inside fit(), fold-safe).
    use_statics : attach static covariates (store/dept/type/size) to series.
    max_samples_per_ts : cap on training windows per series (cost control).
    holiday_weight : per-time-step sample weight on holiday weeks; 5.0 makes
        an L1 training loss exactly the competition WMAE (1.0 disables).
    """

    def __init__(self, stores, features, model_factory, input_chunk_length,
                 output_chunk_length, covariate_preset="none", log_target=True,
                 scale=True, use_statics=True, max_samples_per_ts=None,
                 holiday_weight=5.0, clip_negatives=True):
        self.stores = stores
        self.features = features
        self.model_factory = model_factory
        self.input_chunk_length = int(input_chunk_length)
        self.output_chunk_length = int(output_chunk_length)
        self.covariate_preset = covariate_preset
        self.log_target = log_target
        self.scale = scale
        self.use_statics = use_statics
        self.max_samples_per_ts = max_samples_per_ts
        self.holiday_weight = float(holiday_weight)
        self.clip_negatives = clip_negatives

    def fit(self, raw_train):
        self.cleaner = Cleaner(self.stores, self.features,
                               clip_negatives=self.clip_negatives)
        clean = self.cleaner.fit(raw_train).transform(raw_train)
        statics = (ts_data.build_static_frame(clean, self.stores)
                   if self.use_statics else None)
        self.series_, self.scale_params_, self.report_ = ts_data.build_target_series(
            clean, log_target=self.log_target, scale=self.scale, statics=statics)
        self.train_end_ = pd.to_datetime(clean["Date"]).max()
        self.covs_ = ts_data.build_covariates(self.features, self.covariate_preset)

        need = self.input_chunk_length + self.output_chunk_length
        train_keys = [k for k, s in self.series_.items() if len(s) >= need]
        self.n_train_series_ = len(train_keys)
        self.n_short_series_ = len(self.series_) - len(train_keys)

        fit_kwargs = {}
        if self.covs_ is not None:
            fit_kwargs["future_covariates"] = [self.covs_[k[0]] for k in train_keys]
        if self.max_samples_per_ts is not None:
            fit_kwargs["max_samples_per_ts"] = int(self.max_samples_per_ts)
        if self.holiday_weight != 1.0:
            w = ts_data.build_weight_series(self.features, self.holiday_weight)
            fit_kwargs["sample_weight"] = [w] * len(train_keys)
        self.model_ = self.model_factory()
        self.model_.fit([self.series_[k] for k in train_keys], **fit_kwargs)

        self.fallback_ = _build_fallback(raw_train, self.stores, self.features)
        return self

    def _forecast_scaled(self, keys, n):
        """Deterministic scaled-space forecasts: [(time_index, values)] per key.

        Point models (loss_fn based) predict directly. For likelihood models
        (TFT quantile regression) predict(num_samples=1) would DRAW ONE
        RANDOM SAMPLE, so instead the distribution parameters are predicted
        (deterministic) and the median quantile is kept; horizons beyond
        output_chunk_length use a median-path autoregressive rollout, since
        darts only predicts parameters up to one chunk.
        """
        series_list = [self.series_[k] for k in keys]
        base_kwargs = {}
        if self.covs_ is not None:
            base_kwargs["future_covariates"] = [self.covs_[k[0]] for k in keys]

        if getattr(self.model_, "likelihood", None) is None:
            fcs = self.model_.predict(n=n, series=series_list, num_samples=1,
                                      **base_kwargs)
            return [(fc.time_index, fc.values(copy=False).ravel()) for fc in fcs]

        times = [[] for _ in keys]
        vals = [[] for _ in keys]
        remaining = n
        while remaining > 0:
            step = min(self.output_chunk_length, remaining)
            fcs = self.model_.predict(n=step, series=series_list, num_samples=1,
                                      predict_likelihood_parameters=True,
                                      **base_kwargs)
            for i, fc in enumerate(fcs):
                med = [c for c in fc.components
                       if "_q" in c and abs(float(c.rsplit("_q", 1)[1]) - 0.5) < 1e-9]
                assert len(med) == 1, list(fc.components)
                times[i].append(fc.time_index)
                vals[i].append(fc[med[0]].values(copy=False).ravel())
            remaining -= step
            if remaining > 0:
                series_list = [s.append_values(v[-1].reshape(-1, 1))
                               for s, v in zip(series_list, vals)]

        out = []
        for t_chunks, v_chunks in zip(times, vals):
            idx = t_chunks[0]
            for t in t_chunks[1:]:
                idx = idx.append(t)
            out.append((idx, np.concatenate(v_chunks)))
        return out

    def predict(self, raw_df):
        out = raw_df.copy()
        dates = pd.to_datetime(out["Date"])
        if dates.min() <= self.train_end_:
            raise ValueError("predict() expects dates strictly after the "
                             f"training end {self.train_end_.date()}")
        n = ts_data.weeks_ahead(self.train_end_, dates.max())

        pairs = set(map(tuple, out[["Store", "Dept"]].drop_duplicates().values))
        keys = [k for k in pairs if k in self.series_
                and len(self.series_[k]) >= self.input_chunk_length]
        preds = np.full(len(out), np.nan)
        if keys:
            lookup = {}
            for k, (t_idx, fvals) in zip(keys, self._forecast_scaled(keys, n)):
                vals = fvals.astype(np.float64)
                if self.scale:
                    vmin, vmax = self.scale_params_[k]
                    vals = vals * (vmax - vmin) + vmin
                if self.log_target:
                    vals = np.expm1(vals)
                for d, v in zip(t_idx, vals):
                    lookup[(k[0], k[1], d)] = float(v)
            triples = zip(out["Store"], out["Dept"], dates)
            preds = np.array([lookup.get(t, np.nan) for t in triples])

        missing = np.isnan(preds)
        self.last_fallback_rows_ = int(missing.sum())
        if missing.any():
            preds[missing] = self.fallback_.predict(out.loc[missing])
        assert not np.isnan(preds).any()
        return np.clip(preds, 0.0, None)

    def get_params(self) -> dict:
        return {"pipeline_class": type(self).__name__,
                "input_chunk_length": self.input_chunk_length,
                "output_chunk_length": self.output_chunk_length,
                "covariate_preset": self.covariate_preset,
                "log_target": self.log_target,
                "per_series_minmax_scale": self.scale,
                "use_static_covariates": self.use_statics,
                "max_samples_per_ts": self.max_samples_per_ts,
                "holiday_weight": self.holiday_weight,
                "clip_negatives": self.clip_negatives,
                "fallback": "seasonal_naive_dept_median"}

    # -- persistence: darts torch models need their own save/load, a plain
    # pickle of the Lightning-attached model is not reliable.
    def save(self, dir_path):
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        self.model_.save(str(p / "darts_model.pt"))
        stash = (self.model_, self.model_factory)
        self.model_ = self.model_factory = None
        try:
            joblib.dump(self, p / "wrapper.joblib", compress=3)
        finally:
            self.model_, self.model_factory = stash

    @classmethod
    def load(cls, dir_path):
        from darts.models.forecasting.torch_forecasting_model import (
            TorchForecastingModel)
        p = Path(dir_path)
        pipe = joblib.load(p / "wrapper.joblib")
        try:
            pipe.model_ = TorchForecastingModel.load(str(p / "darts_model.pt"),
                                                     map_location="cpu")
        except TypeError:
            pipe.model_ = TorchForecastingModel.load(str(p / "darts_model.pt"))
        return pipe


def _fit_one(model_factory, series, covs):
    """Fit one local model in a joblib worker; None signals fallback."""
    import logging
    import warnings
    warnings.simplefilter("ignore")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    try:
        model = model_factory()
        if covs is not None:
            model.fit(series, future_covariates=covs)
        else:
            model.fit(series)
        return model
    except Exception:
        return None


class LocalDartsPipeline:
    """Cleaner -> TimeSeries reshaping -> one Darts model PER series.

    model_factory must return a fresh local Darts model (ARIMA, Prophet).
    Series shorter than min_len are not fitted and go to the fallback,
    as do series whose fit raised (non-convergence etc.).
    """

    def __init__(self, stores, features, model_factory, min_len=30,
                 covariate_preset="none", log_target=True,
                 clip_negatives=True, n_jobs=-1):
        self.stores = stores
        self.features = features
        self.model_factory = model_factory
        self.min_len = int(min_len)
        self.covariate_preset = covariate_preset
        self.log_target = log_target
        self.clip_negatives = clip_negatives
        self.n_jobs = n_jobs

    def fit(self, raw_train):
        self.cleaner = Cleaner(self.stores, self.features,
                               clip_negatives=self.clip_negatives)
        clean = self.cleaner.fit(raw_train).transform(raw_train)
        self.series_, _, self.report_ = ts_data.build_target_series(
            clean, log_target=self.log_target, scale=False, statics=None)
        self.train_end_ = pd.to_datetime(clean["Date"]).max()
        self.covs_ = ts_data.build_covariates(self.features, self.covariate_preset)

        eligible = [(k, s) for k, s in self.series_.items()
                    if len(s) >= self.min_len]
        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(self.model_factory, s,
                              self.covs_[k[0]] if self.covs_ is not None else None)
            for k, s in eligible)
        self.models_ = {k: m for (k, _), m in zip(eligible, fitted) if m is not None}
        self.n_fit_failed_ = len(eligible) - len(self.models_)
        self.n_too_short_ = len(self.series_) - len(eligible)

        self.fallback_ = _build_fallback(raw_train, self.stores, self.features)
        return self

    def predict(self, raw_df):
        out = raw_df.copy()
        dates = pd.to_datetime(out["Date"])
        if dates.min() <= self.train_end_:
            raise ValueError("predict() expects dates strictly after the "
                             f"training end {self.train_end_.date()}")
        n = ts_data.weeks_ahead(self.train_end_, dates.max())

        pairs = set(map(tuple, out[["Store", "Dept"]].drop_duplicates().values))
        lookup = {}
        for k in pairs:
            model = self.models_.get(k)
            if model is None:
                continue
            if self.covs_ is not None:
                fc = model.predict(n=n, future_covariates=self.covs_[k[0]])
            else:
                fc = model.predict(n=n)
            vals = fc.values(copy=False).ravel().astype(np.float64)
            if self.log_target:
                vals = np.expm1(vals)
            for d, v in zip(fc.time_index, vals):
                lookup[(k[0], k[1], d)] = float(v)

        triples = zip(out["Store"], out["Dept"], dates)
        preds = np.array([lookup.get(t, np.nan) for t in triples])
        missing = np.isnan(preds)
        self.last_fallback_rows_ = int(missing.sum())
        if missing.any():
            preds[missing] = self.fallback_.predict(out.loc[missing])
        assert not np.isnan(preds).any()
        return np.clip(preds, 0.0, None)

    def get_params(self) -> dict:
        return {"pipeline_class": type(self).__name__,
                "min_len": self.min_len,
                "covariate_preset": self.covariate_preset,
                "log_target": self.log_target,
                "clip_negatives": self.clip_negatives,
                "fallback": "seasonal_naive_dept_median"}

    def save(self, dir_path):
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        stash = self.model_factory
        self.model_factory = None       # factories are lambdas, not picklable
        try:
            joblib.dump(self, p / "wrapper.joblib", compress=3)
        finally:
            self.model_factory = stash

    @classmethod
    def load(cls, dir_path):
        return joblib.load(Path(dir_path) / "wrapper.joblib")


def load_pipeline(dir_path):
    p = Path(dir_path)
    if (p / "darts_model.pt").exists():
        return GlobalDartsPipeline.load(p)
    return LocalDartsPipeline.load(p)


class DartsPipelinePyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper: predict() takes the RAW test frame
    (Store, Dept, Date, IsHoliday), exactly like the tree pipelines."""

    def load_context(self, context):
        from src.ts_pipeline import load_pipeline as _load
        self.pipeline = _load(context.artifacts["pipeline_dir"])

    def predict(self, context, model_input, params=None):
        return self.pipeline.predict(model_input)
