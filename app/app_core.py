"""
Shared application state: data loading, model loading, scoring and explanations,
all cached so the Streamlit pages stay fast and consistent.

A single `get_analysis()` builds the whole analytical object once (pipeline ->
features -> learning table -> monitoring PMI -> scores) and every page reads from
it, so the numbers a practitioner sees are identical across pages.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project importable when run via `streamlit run app/main.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.data_processing import pipeline as dp_pipeline
from src.data_processing import database as db
from src.feature_engineering.features import build_features, MODEL_FEATURES
from src.analytics.monitoring import compute_pmi
from src.ml import dataset as ds
from src.ml import train as ml_train
from src.ml import explain as ml_explain
from src.ml import inference as ml_infer


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Building performance dataset…")
def get_analysis(seed: int = config.RANDOM_SEED) -> dict:
    out = dp_pipeline.run_pipeline(seed=seed, persist=True)
    daily = out["daily"]
    feats = build_features(daily)
    lt = ds.build_learning_table(feats)
    mon = compute_pmi(lt)                       # adds statuses + PMI to the table
    players = out["raw"]["players"]
    episodes = out["raw"]["availability_episodes"]
    matches = out["raw"]["matches"]
    return {
        "daily": daily,
        "features": feats,
        "table": mon,                            # features + target + monitoring
        "weekly": out["weekly"],
        "players": players,
        "episodes": episodes,
        "matches": matches,
        "quality_report": out["quality_report"],
        "clean_stats": out["clean_stats"],
        "split": ds.temporal_split(mon),
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def get_bundle(_train_if_missing: bool = True) -> dict:
    # Try to load a prebuilt bundle. If it is absent OR cannot be unpickled
    # (e.g. a slightly different scikit-learn version in a cloud environment),
    # transparently retrain so the app is always deployable from a clean clone.
    if ml_train.BUNDLE_PATH.exists():
        try:
            return ml_train.load_bundle()
        except Exception:
            pass  # fall through to a fresh, environment-compatible retrain
    if _train_if_missing:
        bundle = ml_train.train_and_select(verbose=False)
        ml_train.save_bundle(bundle)
        return bundle
    raise FileNotFoundError("Model bundle not found. Run `python build_demo.py`.")


@st.cache_resource(show_spinner="Preparing explainer…")
def get_explainer(_bundle_version: str) -> dict:
    bundle = get_bundle()
    return ml_explain.build_explainer(bundle)


@st.cache_data(show_spinner="Scoring squad…")
def get_scores(seed: int = config.RANDOM_SEED, _bundle_version: str = "") -> pd.DataFrame:
    analysis = get_analysis(seed)
    bundle = get_bundle()
    return ml_infer.score_table(bundle, analysis["table"])


def retrain_model() -> dict:
    """Force a retrain (used by the 'Run Model' demo button)."""
    bundle = ml_train.train_and_select(verbose=False)
    ml_train.save_bundle(bundle)
    get_bundle.clear()
    get_explainer.clear()
    get_scores.clear()
    return bundle


# --------------------------------------------------------------------------- #
# Convenience accessors
# --------------------------------------------------------------------------- #
def player_options(players: pd.DataFrame) -> dict[str, str]:
    """Map 'P01 · Name (POS)' -> player_id for selectboxes."""
    return {
        f"{r.player_id} · {r.player_name} ({r.position})": r.player_id
        for r in players.sort_values("player_id").itertuples()
    }


def latest_snapshot(table: pd.DataFrame, scores: pd.DataFrame,
                    players: pd.DataFrame) -> pd.DataFrame:
    """One row per player at the most recent date, with monitoring + risk."""
    last = table.sort_values("date").groupby("player_id").tail(1)
    last_scores = scores.sort_values("date").groupby("player_id").tail(1)
    snap = last.merge(last_scores[["player_id", "risk_probability", "monitoring_level"]],
                      on="player_id", how="left")
    snap = snap.merge(players[["player_id", "player_name", "position", "age"]],
                      on="player_id", how="left", suffixes=("", "_p"))
    if "player_name" not in snap or snap["player_name"].isna().all():
        snap["player_name"] = snap["player_id"]
    return snap
