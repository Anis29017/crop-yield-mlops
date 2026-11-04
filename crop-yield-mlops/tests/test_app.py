"""Unit tests: data prep, model loading/training, and API endpoints."""

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app import app, load_model
from src.prepare_data import (
    REQUIRED_COLS,
    TARGET,
    make_synthetic,
    prepare_data,
    validate,
)
from src.train import split

VALID_INPUT = {
    "district": "Faisalabad",
    "province": "Punjab",
    "crop": "wheat",
    "year": 2019,
    "season_tmax_mean": 33.5,
    "season_rain_mm": 240.0,
    "heat_days_40c": 14,
    "rain_anomaly": -10.0,
    "soil_ph": 8.1,
    "soil_oc": 5.2,
    "soil_clay": 22.0,
    "soil_sand": 41.0,
    "soil_nitrogen": 0.7,
    "prev_yield": 3.1,
}

client = TestClient(app)


# ---------- data preparation ----------


def test_prepare_data(tmp_path):
    out = tmp_path / "processed.csv"
    prepare_data(raw_path=str(tmp_path / "missing.csv"), out_path=str(out))
    df = pd.read_csv(out)
    assert len(df) > 0
    assert df.isnull().sum().sum() == 0
    assert list(df.columns) == REQUIRED_COLS


def test_validate_rejects_missing_values():
    df = make_synthetic(n_districts=2)
    df.loc[0, "soil_ph"] = None
    with pytest.raises(AssertionError, match="Missing values"):
        validate(df)


def test_validate_rejects_duplicate_rows():
    df = make_synthetic(n_districts=2)
    df = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(AssertionError, match="Duplicate"):
        validate(df)


def test_validate_rejects_implausible_yield():
    df = make_synthetic(n_districts=2)
    df.loc[df["crop"] == "wheat", TARGET] = 50.0  # wheat cannot be 50 t/ha
    with pytest.raises(AssertionError, match="wheat"):
        validate(df)


def test_time_split_has_no_future_leakage():
    df = make_synthetic(n_districts=2)
    train, val = split(df)
    assert train["year"].max() <= 2017
    assert val["year"].min() >= 2018 and val["year"].max() <= 2020
    assert train["year"].max() < val["year"].min()


# ---------- model ----------


@pytest.mark.skipif(not os.path.exists("models/model.pkl"), reason="run train.py first")
def test_model_loads_and_predicts_positive_yield():
    model = load_model()
    row = pd.DataFrame([VALID_INPUT]).drop(columns="district")
    pred = model.predict(row)
    assert len(pred) == 1
    assert pred[0] > 0


# ---------- API ----------


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.skipif(not os.path.exists("models/model.pkl"), reason="run train.py first")
def test_predict_valid_input():
    r = client.post("/predict", json=VALID_INPUT)
    assert r.status_code == 200
    body = r.json()
    assert body["crop"] == "wheat"
    assert body["predicted_yield_t_ha"] > 0


def test_predict_invalid_type():
    bad = {**VALID_INPUT, "season_rain_mm": "not_a_number"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_unknown_crop():
    bad = {**VALID_INPUT, "crop": "banana"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_missing_field():
    bad = {k: v for k, v in VALID_INPUT.items() if k != "prev_yield"}
    assert client.post("/predict", json=bad).status_code == 422
