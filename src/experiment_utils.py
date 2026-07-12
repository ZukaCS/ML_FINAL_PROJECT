"""MLflow helpers shared by every model notebook, so both teammates produce
identical run shapes, metric keys and artifacts.

Run structure per architecture X :
    experiment "X_Training"
        run "X_Preprocessing"        <- one child per cleaning config
        run "X_Feature_Engineering"  <- one child per feature-group set
        run "X_Feature_Selection"    <- one child per candidate feature subset
        run "X_CV"                   <- one child per attempt (Optuna trial or fixed config)
        run "X_Final"                <- retrained best config + saved pipeline
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd


def _dagshub_creds():
    """Colab Secrets (DAGSHUB_USER / DAGSHUB_TOKEN) first, else prompt."""
    try:
        from google.colab import userdata
        return userdata.get("DAGSHUB_USER"), userdata.get("DAGSHUB_TOKEN")
    except Exception:
        pass  # not on Colab, or secrets not set -> ask
    import getpass
    user = input("DagsHub username (leave blank to log locally): ").strip()
    if not user:
        return None, None
    return user, getpass.getpass("DagsHub token: ").strip()


def setup_mlflow(root, dagshub_repo: str | None = None) -> str:
    """Resolve the tracking destination, in order:

    1. MLFLOW_TRACKING_URI env var (session already configured)
    2. dagshub_repo ("owner/repo") + credentials from Colab Secrets or an
       interactive prompt
    3. local ./mlruns fallback (solo iteration only)
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri and dagshub_repo:
        user, token = _dagshub_creds()
        if user and token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            uri = f"https://dagshub.com/{dagshub_repo}.mlflow"
            os.environ["MLFLOW_TRACKING_URI"] = uri
    if not uri:
        # newer MLflow raises on the filesystem backend unless opted in
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        uri = f"file:{Path(root) / 'mlruns'}"
    mlflow.set_tracking_uri(uri)
    return mlflow.get_tracking_uri()


def _loggable(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple, set, dict)):
            v = json.dumps(list(v) if not isinstance(v, dict) else v)
        out[k] = str(v)[:500]  # MLflow param value limit
    return out


def log_table(df: pd.DataFrame, filename: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / filename
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))


def run_stage(model_name: str, stage: str, configs: dict, build_and_eval,
              parent_log=None):
    """One parent MLflow run per stage, one nested child per config.

    configs        : {config_name: cfg_dict}
    build_and_eval : cfg_dict -> metrics dict (from validation.evaluate)
    parent_log     : optional callback logging extra artifacts on the parent
                     run (e.g. a feature-importance figure)

    Returns (results df sorted by wmae_mean, name of best config).
    """
    mlflow.set_experiment(f"{model_name}_Training")
    rows = []
    with mlflow.start_run(run_name=f"{model_name}_{stage}"):
        mlflow.set_tag("stage", stage)
        if parent_log is not None:
            parent_log()
        for name, cfg in configs.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params(_loggable(cfg))
                metrics = build_and_eval(cfg)
                mlflow.log_metrics(metrics)
                rows.append({"config": name, **metrics})
                print(f"[{stage}] {name}: "
                      + ", ".join(f"{k}={v:,.0f}" for k, v in metrics.items()))
        results = (pd.DataFrame(rows).set_index("config")
                   .sort_values("wmae_mean"))
        log_table(results.reset_index(), f"{stage}_comparison.csv")
        best = results.index[0]
        mlflow.set_tag("best_config", best)
        mlflow.log_metric("best_wmae_mean", float(results["wmae_mean"].iloc[0]))
        # winner's settings on the parent, so the stage outcome reads standalone
        mlflow.log_params({f"best_{k}": v
                           for k, v in _loggable(configs[best]).items()})
    return results, best


class PyfuncPipeline(mlflow.pyfunc.PythonModel):
    """Wraps a fitted WalmartPipeline so MLflow can serve/load it as pyfunc.
    predict() takes the RAW test frame (Store, Dept, Date, IsHoliday)."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input, params=None):
        return self.pipeline.predict(model_input)
