"""
Build the supervised learning dataset with a leakage-safe target and a temporal
split.

Task framing
------------
On a given monitoring day t (when the player is currently AVAILABLE), predict
whether the player will enter a reduced-availability state (modified training OR
unavailable) within the next H days (config.PREDICTION_HORIZON_DAYS).

Two decisions matter:

* We only keep rows where the player is currently available, so the model learns
  to anticipate a *transition* rather than trivially detecting that an ongoing
  episode continues.
* Every feature is causal (built from data up to day t). The target uses only
  future availability labels. Feature columns and label columns never mix — the
  leakage blocklist in config is asserted here.

The split is strictly temporal (no shuffling): the earliest part of the season
trains the model, the middle validates/calibrates it, and the most recent part
tests it. This mirrors deployment, where you always predict the future from the
past.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.feature_engineering.features import MODEL_FEATURES


def _forward_event(g: pd.Series, horizon: int) -> pd.Series:
    """1 if an event (value==1) occurs within the next `horizon` days (t+1..t+H)."""
    # reverse rolling sum over the next H days, excluding today.
    rev = g[::-1]
    fwd = rev.rolling(horizon, min_periods=1).sum().shift(1)[::-1]
    return (fwd.fillna(0) > 0).astype(int)


def build_learning_table(features: pd.DataFrame,
                         horizon: int = config.PREDICTION_HORIZON_DAYS) -> pd.DataFrame:
    """Attach the target and the modelling mask to the feature table."""
    df = features.sort_values(["player_id", "date"]).copy()

    df["is_event_day"] = (df["availability_status"] != "available").astype(int)
    df["target"] = (
        df.groupby("player_id", group_keys=False)["is_event_day"]
        .apply(lambda s: _forward_event(s, horizon))
    )

    # Feature warm-up: require ~4 weeks of history so rolling/baseline features
    # are stable.
    df["season_day"] = (df["date"] - df["date"].min()).dt.days
    warmed = df["season_day"] >= 28

    # Only predict from days where the player is currently available.
    currently_available = df["availability_status"] == "available"

    # Label must be fully observable: drop the last H days of the timeline.
    max_day = df["season_day"].max()
    label_observed = df["season_day"] <= (max_day - horizon)

    df["in_model"] = (warmed & currently_available & label_observed).astype(int)
    return df


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split modelling rows into train/valid/test by calendar time."""
    model_rows = df[df["in_model"] == 1].copy()
    day = model_rows["season_day"]
    max_day = day.max()
    train_end = max_day * config.TRAIN_FRACTION
    valid_end = max_day * (config.TRAIN_FRACTION + config.VALID_FRACTION)

    train = model_rows[day <= train_end]
    valid = model_rows[(day > train_end) & (day <= valid_end)]
    test = model_rows[day > valid_end]
    return {"train": train, "valid": valid, "test": test,
            "cut_train": train_end, "cut_valid": valid_end}


def assert_no_leakage(feature_cols: list[str]) -> None:
    """Guard: no blocklisted (post-outcome) column may be a model feature."""
    leaks = set(feature_cols) & set(config.LEAKAGE_BLOCKLIST)
    if leaks:
        raise ValueError(f"Data leakage: blocklisted columns used as features: {leaks}")


def get_xy(part: pd.DataFrame, feature_cols: list[str] | None = None):
    feature_cols = feature_cols or MODEL_FEATURES
    assert_no_leakage(feature_cols)
    X = part[feature_cols].astype(float).fillna(0.0)
    y = part["target"].astype(int)
    return X, y


if __name__ == "__main__":
    from src.data_processing import pipeline
    from src.feature_engineering.features import build_features

    out = pipeline.run_pipeline(persist=False)
    feats = build_features(out["daily"])
    lt = build_learning_table(feats)
    sp = temporal_split(lt)
    for name in ("train", "valid", "test"):
        part = sp[name]
        print(f"{name:6s} n={len(part):5d}  positives={int(part['target'].sum()):4d}"
              f"  rate={part['target'].mean():.3f}")
