"""
Inference layer: score the feature table and assign monitoring levels.

Keeps a single source of truth for how a calibrated probability becomes an
operational LOW / MODERATE / HIGH monitoring level, using the bands stored in
the model bundle (derived from the training base rate and the operating
threshold). The same mapping is used everywhere in the app so the number a
practitioner sees always means the same thing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml import dataset as ds


def monitoring_level(prob: float, level_bands: dict) -> str:
    if prob >= level_bands["high"]:
        return "HIGH"
    if prob >= level_bands["moderate"]:
        return "MODERATE"
    return "LOW"


def score_table(bundle: dict, features_with_target: pd.DataFrame) -> pd.DataFrame:
    """Return per-row calibrated risk probability + monitoring level.

    Scores every row (not only the modelling mask) so the app can show a risk
    trajectory across the whole season for a player. Rows are still built from
    causal features, so this is a legitimate 'what would the model have said on
    that day' backtest.
    """
    model = bundle["calibrated_model"]
    cols = bundle["feature_cols"]
    X = features_with_target[cols].astype(float).fillna(0.0)
    proba = model.predict_proba(X)[:, 1]

    out = features_with_target[["player_id", "date"]].copy()
    out["risk_probability"] = proba
    out["monitoring_level"] = [monitoring_level(p, bundle["level_bands"]) for p in proba]
    return out


def confidence_from_probability(prob: float) -> float:
    """A simple, honest 'confidence' proxy: how far the probability sits from
    the 50/50 decision boundary, rescaled to 0-100. This is a communication aid
    for staff, NOT a statistical confidence interval (see README limitations).
    """
    return round(float(abs(prob - 0.5) * 2) * 100, 0)


def latest_scores(bundle: dict, features_with_target: pd.DataFrame) -> pd.DataFrame:
    """Most recent scored row per player (the squad's 'today')."""
    scored = score_table(bundle, features_with_target)
    scored = scored.sort_values("date").groupby("player_id").tail(1)
    return scored.reset_index(drop=True)
