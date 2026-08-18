"""Chargement des artefacts et préparation des scénarios utilisateur."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from modeling import predict_kbtu


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def load_artifacts() -> tuple[object, object, dict]:
    without_score = joblib.load(ARTIFACTS / "without_energy_star.joblib")
    with_score = joblib.load(ARTIFACTS / "with_energy_star.joblib")
    metadata = json.loads((ARTIFACTS / "demo_metadata.json").read_text(encoding="utf-8"))
    return without_score, with_score, metadata


def build_scenario(profile: dict, updates: dict) -> pd.DataFrame:
    features = profile["features"].copy()
    features.update(updates)
    return pd.DataFrame([features])


def predict_scenario(without_score, with_score, scenario: pd.DataFrame) -> dict[str, float]:
    return {
        "without_energy_star_kbtu": predict_kbtu(without_score, scenario),
        "with_energy_star_kbtu": predict_kbtu(with_score, scenario),
    }

