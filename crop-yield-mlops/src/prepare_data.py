"""Data preparation step (separate from training, as required for M1).

Reads the merged master dataset (district x year x crop) if it exists,
otherwise generates a synthetic placeholder with the SAME schema so the
rest of the pipeline (training, API, tests) can be built and tested
before the real data is ready.

Output: data/processed.csv
"""

import os

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/master.csv"
OUT_PATH = "data/processed.csv"

CROPS = ["wheat", "cotton", "rice", "sugarcane"]
PROVINCES = ["Punjab", "Sindh", "KP", "Balochistan"]

FEATURES = [
    "crop",
    "province",
    "year",
    "season_tmax_mean",  # mean daily max temp over the crop's season (C)
    "season_rain_mm",  # total rainfall over the crop's season (mm)
    "heat_days_40c",  # days above 40C in the season
    "rain_anomaly",  # season rain minus district long-term mean (mm)
    "soil_ph",
    "soil_oc",  # organic carbon (g/kg)
    "soil_clay",  # %
    "soil_sand",  # %
    "soil_nitrogen",  # g/kg
    "prev_yield",  # previous year's yield, same district & crop (t/ha)
]
TARGET = "yield_t_ha"
ID_COLS = ["district"]
REQUIRED_COLS = ID_COLS + FEATURES + [TARGET]

# Plausible yield ranges (t/ha) used by validation to catch extraction errors.
YIELD_BOUNDS = {
    "wheat": (0.3, 6.0),
    "cotton": (0.1, 4.0),
    "rice": (0.3, 6.0),
    "sugarcane": (10.0, 120.0),
}

# Typical mean yields (t/ha), used only by the synthetic generator.
_BASE_YIELD = {"wheat": 3.0, "cotton": 0.8, "rice": 2.5, "sugarcane": 60.0}


def make_synthetic(n_districts=40, start_year=2000, end_year=2023, seed=42):
    """Synthetic placeholder with the real schema. Includes a 2010 and 2022
    flood shock so the drift experiment in M2 has something to detect."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_districts):
        district = f"district_{d:02d}"
        province = PROVINCES[d % len(PROVINCES)]
        soil = {
            "soil_ph": rng.uniform(7.5, 8.6),
            "soil_oc": rng.uniform(2.0, 9.0),
            "soil_clay": rng.uniform(10, 40),
            "soil_sand": rng.uniform(20, 60),
            "soil_nitrogen": rng.uniform(0.3, 1.2),
        }
        for crop in CROPS:
            prev = _BASE_YIELD[crop]
            for year in range(start_year, end_year + 1):
                shock = year in (2010, 2022)
                tmax = rng.normal(34, 1.5) + year * 0.02 - 40
                rain = rng.normal(250, 60) + (400 if shock else 0)
                heat = max(0, int(rng.normal(12, 4) + (year - 2000) * 0.2))
                anomaly = rain - 250
                effect = (
                    1
                    - 0.01 * heat
                    - (0.35 if shock else 0)
                    + 0.02 * soil["soil_oc"] / 9
                    + rng.normal(0, 0.05)
                )
                yld = max(_BASE_YIELD[crop] * effect, YIELD_BOUNDS[crop][0] + 0.01)
                rows.append(
                    {
                        "district": district,
                        "province": province,
                        "crop": crop,
                        "year": year,
                        "season_tmax_mean": round(tmax + 40, 2),
                        "season_rain_mm": round(max(rain, 0), 1),
                        "heat_days_40c": heat,
                        "rain_anomaly": round(anomaly, 1),
                        **{k: round(v, 3) for k, v in soil.items()},
                        "prev_yield": round(prev, 3),
                        TARGET: round(yld, 3),
                    }
                )
                prev = yld
    return pd.DataFrame(rows)


def validate(df):
    """Basic validation rules. Raises AssertionError with a clear message."""
    missing = set(REQUIRED_COLS) - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert len(df) > 0, "Empty dataset"
    assert df[REQUIRED_COLS].isnull().sum().sum() == 0, "Missing values found"
    assert not df.duplicated(["district", "year", "crop"]).any(), (
        "Duplicate district x year x crop rows"
    )
    assert set(df["crop"]) <= set(CROPS), f"Unknown crop in {set(df['crop'])}"
    for crop, (lo, hi) in YIELD_BOUNDS.items():
        y = df.loc[df["crop"] == crop, TARGET]
        bad = y[(y < lo) | (y > hi)]
        assert bad.empty, f"{len(bad)} {crop} yields outside [{lo}, {hi}] t/ha"
    return True


def prepare_data(raw_path=RAW_PATH, out_path=OUT_PATH):
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        source = raw_path
    else:
        df = make_synthetic()
        source = "synthetic placeholder"
    df = df[REQUIRED_COLS]
    validate(df)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Data ready from {source}: {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    prepare_data()
