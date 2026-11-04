"""Training with MLflow tracking.

- Time-based split: train <= 2017, validate 2018-2020, test 2021-2023
  (test is NOT used here; it is reserved for the M2 drift experiment).
- Logs a last-year baseline plus RF / XGBoost / LightGBM runs with
  different hyperparameters.
- Saves the best model (lowest validation RMSE) to models/model.pkl.
"""

import json
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.prepare_data import CROPS, FEATURES, PROVINCES, TARGET

DATA_PATH = "data/processed.csv"
MODEL_PATH = "models/model.pkl"
BEST_RUN_PATH = "models/best_run.json"
EXPERIMENT = "crop-yield-m1"
REGISTERED_NAME = "crop-yield-model"

TRAIN_END = 2017
VAL_START, VAL_END = 2018, 2020

CATEGORICAL = ["crop", "province"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


def split(df):
    train = df[df["year"] <= TRAIN_END]
    val = df[(df["year"] >= VAL_START) & (df["year"] <= VAL_END)]
    return train, val


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred, crops=None):
    """Pooled metrics, plus per-crop RMSE when `crops` is given. Per-crop RMSE
    matters because sugarcane (~60 t/ha) dominates a pooled RMSE."""
    out = {
        "val_rmse": rmse(y_true, y_pred),
        "val_mae": float(mean_absolute_error(y_true, y_pred)),
        "val_r2": float(r2_score(y_true, y_pred)),
    }
    if crops is not None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        crops = np.asarray(crops)
        for crop in CROPS:
            mask = crops == crop
            if mask.any():
                out[f"val_rmse_{crop}"] = rmse(y_true[mask], y_pred[mask])
    return out


def build_pipeline(model):
    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(categories=[CROPS, PROVINCES], handle_unknown="ignore"),
                CATEGORICAL,
            ),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


# Each entry = one MLflow run. 9 model runs + 1 baseline = 10 runs.
CANDIDATES = [
    ("random_forest", RandomForestRegressor, {"n_estimators": 100, "max_depth": 6}),
    ("random_forest", RandomForestRegressor, {"n_estimators": 300, "max_depth": 10}),
    ("random_forest", RandomForestRegressor, {"n_estimators": 300, "max_depth": None}),
    (
        "xgboost",
        XGBRegressor,
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
    ),
    (
        "xgboost",
        XGBRegressor,
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05},
    ),
    (
        "xgboost",
        XGBRegressor,
        {"n_estimators": 600, "max_depth": 7, "learning_rate": 0.03},
    ),
    (
        "lightgbm",
        LGBMRegressor,
        {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.1},
    ),
    (
        "lightgbm",
        LGBMRegressor,
        {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.05},
    ),
    (
        "lightgbm",
        LGBMRegressor,
        {"n_estimators": 600, "num_leaves": 63, "learning_rate": 0.03},
    ),
]


def log_baseline(val):
    """Naive baseline: predict last year's yield. M1 exit criterion is to beat this."""
    with mlflow.start_run(run_name="baseline_last_year"):
        mlflow.log_param("model_type", "baseline_last_year")
        m = metrics(val[TARGET], val["prev_yield"], val["crop"])
        mlflow.log_metrics(m)
        print(f"baseline_last_year  val_rmse={m['val_rmse']:.4f}")
        return m["val_rmse"]


def train_one(name, cls, params, train, val):
    extra = {"random_state": 42}
    if name == "lightgbm":
        extra["verbose"] = -1
    pipe = build_pipeline(cls(**params, **extra))
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_params(params)
        pipe.fit(train[FEATURES], train[TARGET])
        m = metrics(val[TARGET], pipe.predict(val[FEATURES]), val["crop"])
        mlflow.log_metrics(m)
        # cloudpickle: we trained this model ourselves, so it is a trusted source
        # (newer MLflow defaults to skops, which rejects tree models).
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        print(f"{name:14s} {params}  val_rmse={m['val_rmse']:.4f}")
        return pipe, m, run.info.run_id


def train(data_path=DATA_PATH, candidates=CANDIDATES):
    df = pd.read_csv(data_path)
    train_df, val_df = split(df)
    mlflow.set_experiment(EXPERIMENT)

    baseline_rmse = log_baseline(val_df)
    best = None
    for name, cls, params in candidates:
        pipe, m, run_id = train_one(name, cls, params, train_df, val_df)
        if best is None or m["val_rmse"] < best["metrics"]["val_rmse"]:
            best = {
                "pipe": pipe,
                "metrics": m,
                "run_id": run_id,
                "model_type": name,
                "params": params,
            }

    # Register the best run's model in the MLflow Model Registry (M1 deliverable).
    version = mlflow.register_model(f"runs:/{best['run_id']}/model", REGISTERED_NAME)
    print(f"Registered '{REGISTERED_NAME}' version {version.version}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best["pipe"], f)

    summary = {
        "run_id": best["run_id"],
        "model_type": best["model_type"],
        "params": best["params"],
        **best["metrics"],
        "baseline_val_rmse": baseline_rmse,
        "beats_baseline": best["metrics"]["val_rmse"] < baseline_rmse,
    }
    with open(BEST_RUN_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Best run:", json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    train()
