"""Shared evaluation: the WMAE metric + fixed rolling-origin folds.

Both teammates import THIS in every notebook — if folds or the metric
diverge between us, the cross-architecture comparison stops meaning
anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FOLDS = [
    # rehearses Thanksgiving + Christmas — the fold that predicts the leaderboard
    {"name": "fold1_holiday", "train_end": "2011-10-28",
     "val_start": "2011-11-04", "val_end": "2012-01-27"},
    {"name": "fold2_spring", "train_end": "2012-02-03",
     "val_start": "2012-02-10", "val_end": "2012-04-27"},
    {"name": "fold3_recent", "train_end": "2012-07-27",
     "val_start": "2012-08-03", "val_end": "2012-10-26"},
]


def wmae(y_true, y_pred, is_holiday) -> float:
    """Competition metric: MAE with 5x weight on holiday weeks."""
    w = np.where(np.asarray(is_holiday).astype(bool), 5.0, 1.0)
    err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    return float((w * err).sum() / w.sum())


def split_fold(train: pd.DataFrame, fold: dict):
    dates = pd.to_datetime(train["Date"])
    tr = train[dates <= pd.Timestamp(fold["train_end"])]
    va = train[(dates >= pd.Timestamp(fold["val_start"]))
               & (dates <= pd.Timestamp(fold["val_end"]))]
    return tr, va


def evaluate(pipeline_factory, train: pd.DataFrame, folds=FOLDS,
             train_metrics: bool = False) -> dict:
    """Fit a FRESH pipeline per fold (so fit-time state never sees the
    validation window) and score WMAE.

    pipeline_factory: () -> object with .fit(raw_df) / .predict(raw_df).
    Returns {"wmae_fold1": ..., "wmae_fold2": ..., "wmae_fold3": ...,
             "wmae_mean": ...} — these metric names are the shared
    convention across all MLflow experiments; do not rename.

    train_metrics=True additionally scores each fold's own training data
    and the gap (val - train), the overfitting signal:
    train_wmae_fold{i}, gap_fold{i}, train_wmae_mean, gap_mean.
    """
    val, tr_scores = [], []
    out = {}
    for i, fold in enumerate(folds, start=1):
        tr, va = split_fold(train, fold)
        pipe = pipeline_factory().fit(tr)
        # drop the target so validation looks exactly like the raw test set
        pred = pipe.predict(va.drop(columns=["Weekly_Sales"]))
        v = wmae(va["Weekly_Sales"], pred, va["IsHoliday"])
        out[f"wmae_fold{i}"] = v
        val.append(v)
        if train_metrics:
            pred_tr = pipe.predict(tr.drop(columns=["Weekly_Sales"]))
            t = wmae(tr["Weekly_Sales"], pred_tr, tr["IsHoliday"])
            out[f"train_wmae_fold{i}"] = t
            out[f"gap_fold{i}"] = v - t
            tr_scores.append(t)
    out["wmae_mean"] = float(np.mean(val))
    if train_metrics:
        out["train_wmae_mean"] = float(np.mean(tr_scores))
        out["gap_mean"] = out["wmae_mean"] - out["train_wmae_mean"]
    return out
