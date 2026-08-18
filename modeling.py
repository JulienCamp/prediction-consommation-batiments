"""Prétraitement et modèles du démonstrateur de consommation énergétique."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PowerTransformer, StandardScaler


RANDOM_STATE = 37
TARGET = "SiteEnergyUseWN(kBtu)"
KBTU_TO_MWH = 0.00029307107


class CenterDistance(BaseEstimator, TransformerMixin):
    """Ajoute la distance au centre médian appris, en mètres."""

    def fit(self, X: pd.DataFrame, y=None):
        self.center_latitude_ = float(X["Latitude"].median())
        self.center_longitude_ = float(X["Longitude"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        lat1 = np.radians(result["Latitude"].astype(float).to_numpy())
        lon1 = np.radians(result["Longitude"].astype(float).to_numpy())
        lat2 = np.radians(self.center_latitude_)
        lon2 = np.radians(self.center_longitude_)
        dlat = lat1 - lat2
        dlon = lon1 - lon2
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        result["DistanceToCenter"] = 6_371_000 * 2 * np.arcsin(np.sqrt(a))
        return result.drop(columns=["Latitude", "Longitude"])


def make_preprocessor(include_energy_star: bool) -> Pipeline:
    transformers: list[tuple] = [
        (
            "passthrough_numeric",
            "passthrough",
            [
                "NumberofBuildings",
                "NumberofPropertyUseTypes",
                "CouncilDistrictCode",
                "SteamProportion",
                "DistanceToCenter",
            ],
        ),
        (
            "log_transformer",
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ["PropertyGFABuilding(s)"],
        ),
        (
            "power_transformer",
            PowerTransformer(),
            ["NumberofFloors", "PropertyAge", "NaturalGasProportion", "BuildingsProportion"],
        ),
        (
            "onehot_categorical",
            OneHotEncoder(handle_unknown="ignore", min_frequency=0.02, sparse_output=False),
            ["BuildingType", "PrimaryPropertyType"],
        ),
    ]
    if include_energy_star:
        transformers.append(("energy_star_imputation", KNNImputer(n_neighbors=3), ["ENERGYSTARScore"]))

    return Pipeline(
        [
            ("add_distance", CenterDistance()),
            ("columns", ColumnTransformer(transformers, remainder="drop")),
            ("scale", StandardScaler()),
        ]
    )


def make_random_forest(include_energy_star: bool, n_estimators: int = 300) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(include_energy_star)),
            (
                "regressor",
                RandomForestRegressor(
                    max_depth=20 if include_energy_star else None,
                    n_estimators=n_estimators,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def predict_kbtu(model: Pipeline, building: pd.DataFrame) -> float:
    prediction_log = float(model.predict(building)[0])
    return max(float(np.expm1(prediction_log)), 0.0)
