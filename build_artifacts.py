"""Entraîne les deux modèles et exporte trois profils de démonstration."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from modeling import RANDOM_STATE, TARGET, make_random_forest


ROOT = Path(__file__).resolve().parent
SOURCE_DATA = ROOT.parents[1] / "A deposer - anciens projets" / "Projet 4 - Bâtiments" / "cleaned_data.csv"
ARTIFACTS = ROOT / "artifacts"


def metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, predictions)),
        "mae_kbtu": float(mean_absolute_error(y_true, predictions)),
        "rmse_kbtu": float(mean_squared_error(y_true, predictions) ** 0.5),
        "median_absolute_error_kbtu": float(np.median(np.abs(y_true.to_numpy() - predictions))),
    }


def json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    data = pd.read_csv(SOURCE_DATA, index_col=0)
    X = data.drop(columns=[TARGET, "TotalGHGEmissions"])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=data["BuildingType"],
    )

    models = {}
    results = {}
    for key, include_score in (("without_energy_star", False), ("with_energy_star", True)):
        model = make_random_forest(include_score)
        model.fit(X_train, np.log1p(y_train))
        predictions = np.maximum(np.expm1(model.predict(X_test)), 0)
        models[key] = model
        results[key] = metrics(y_test, predictions)
        joblib.dump(model, ARTIFACTS / f"{key}.joblib", compress=3)

    quantiles = [("Petit bâtiment", 0.20), ("Bâtiment intermédiaire", 0.50), ("Grand bâtiment énergivore", 0.85)]
    available = y_test.copy()
    chosen = []
    for label, quantile in quantiles:
        target_value = float(y_test.quantile(quantile))
        index = (available - target_value).abs().idxmin()
        chosen.append((label, index))
        available = available.drop(index)

    energy_star_median = float(X_train["ENERGYSTARScore"].median())
    profiles = []
    for label, index in chosen:
        row = X_test.loc[index].copy()
        score_was_missing = pd.isna(row["ENERGYSTARScore"])
        if score_was_missing:
            row["ENERGYSTARScore"] = energy_star_median
        record = {column: json_value(value) for column, value in row.items()}
        profiles.append(
            {
                "label": label,
                "description": f"{record['PrimaryPropertyType']} · {int(record['PropertyGFABuilding(s)']):,} pi²".replace(",", " "),
                "source_energy_star_missing": bool(score_was_missing),
                "observed_kbtu": float(y_test.loc[index]),
                "features": record,
            }
        )

    metadata = {
        "training": {
            "rows": len(data),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "split": "70/30, random_state=37, stratification par BuildingType",
            "target": TARGET,
            "n_estimators": 300,
        },
        "metrics": results,
        "benchmarks_kbtu": {
            "q25": float(y_test.quantile(0.25)),
            "median": float(y_test.median()),
            "q75": float(y_test.quantile(0.75)),
        },
        "categories": {
            "building_types": sorted(X_train["BuildingType"].dropna().unique().tolist()),
            "property_types": sorted(X_train["PrimaryPropertyType"].dropna().unique().tolist()),
        },
        "profiles": profiles,
    }
    (ARTIFACTS / "demo_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
