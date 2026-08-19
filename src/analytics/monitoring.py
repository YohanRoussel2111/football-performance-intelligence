"""
Multidimensional player monitoring.

Rather than collapse everything into one opaque "risk number", we expose several
interpretable dimensions, each derived from documented sports-science heuristics:

    Load Status         : Normal / Elevated / High
    Recovery Status     : Good / Moderate / Poor
    Neuromuscular Status: Stable / Reduced / Critical
    Match Congestion    : Low / Moderate / High
    Availability Status : Available / Monitor / Modified / Unavailable

These feed a transparent Performance Monitoring Index (PMI, 0-100) whose
component contributions are always available, so a practitioner can see *which*
dimension is driving a flag. The PMI is a monitoring/triage aid, not a diagnosis.

All thresholds live in `src.config.THRESHOLDS` and are documented in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

T = config.THRESHOLDS


# --------------------------------------------------------------------------- #
# Dimensional classifiers (row-wise, vectorized)
# --------------------------------------------------------------------------- #
def load_status(df: pd.DataFrame) -> pd.Series:
    acwr = df["acwr"]
    spike = df["load_week_change"]
    out = np.select(
        [
            (acwr >= T.acwr_very_high) | (spike > 0.6),
            (acwr >= T.acwr_high) | (spike > 0.3) | (df["monotony"] > T.monotony_high),
        ],
        ["High", "Elevated"],
        default="Normal",
    )
    return pd.Series(out, index=df.index)


def recovery_status(df: pd.DataFrame) -> pd.Series:
    # Worse recovery => lower sleep quality, higher fatigue/soreness (z-scores).
    burden = (-df["sleep_quality_z"] + df["fatigue_z"] + df["muscle_soreness_z"]) / 3.0
    out = np.select(
        [burden >= 1.0, burden >= 0.5],
        ["Poor", "Moderate"],
        default="Good",
    )
    return pd.Series(out, index=df.index)


def neuromuscular_status(df: pd.DataFrame) -> pd.Series:
    drop = df["cmj_pct_change"]
    z = df["cmj_z"]
    out = np.select(
        [(drop <= T.cmj_drop_critical) | (z <= -2.0),
         (drop <= T.cmj_drop_reduced) | (z <= -1.0)],
        ["Critical", "Reduced"],
        default="Stable",
    )
    return pd.Series(out, index=df.index)


def congestion_status(df: pd.DataFrame) -> pd.Series:
    out = np.select(
        [
            (df["minutes_7d"] >= T.congestion_minutes_7d_high) |
            (df["matches_14d"] >= T.congestion_matches_14d_high),
            (df["minutes_7d"] >= 120) | (df["matches_14d"] >= 2),
        ],
        ["High", "Moderate"],
        default="Low",
    )
    return pd.Series(out, index=df.index)


# --------------------------------------------------------------------------- #
# Performance Monitoring Index
# --------------------------------------------------------------------------- #
# Each dimension is mapped to a 0-100 "attention" sub-score (higher = more
# attention warranted), then combined with documented weights.
_LOAD_SCORE = {"Normal": 10, "Elevated": 55, "High": 90}
_RECOVERY_SCORE = {"Good": 10, "Moderate": 55, "Poor": 90}
_NM_SCORE = {"Stable": 10, "Reduced": 60, "Critical": 95}
_CONG_SCORE = {"Low": 10, "Moderate": 50, "High": 85}


def compute_pmi(df: pd.DataFrame) -> pd.DataFrame:
    """Attach dimensional statuses, PMI and per-dimension contributions."""
    out = df.copy()
    out["load_status"] = load_status(out)
    out["recovery_status"] = recovery_status(out)
    out["neuromuscular_status"] = neuromuscular_status(out)
    out["congestion_status"] = congestion_status(out)

    w = config.PMI_WEIGHTS
    load_c = out["load_status"].map(_LOAD_SCORE) * w["load"]
    recov_c = out["recovery_status"].map(_RECOVERY_SCORE) * w["recovery"]
    nm_c = out["neuromuscular_status"].map(_NM_SCORE) * w["neuromuscular"]
    cong_c = out["congestion_status"].map(_CONG_SCORE) * w["congestion"]

    out["pmi_load_contrib"] = load_c
    out["pmi_recovery_contrib"] = recov_c
    out["pmi_neuromuscular_contrib"] = nm_c
    out["pmi_congestion_contrib"] = cong_c
    out["pmi"] = (load_c + recov_c + nm_c + cong_c).round(1)

    # Human-facing monitoring band derived from the PMI.
    out["pmi_band"] = pd.cut(
        out["pmi"], bins=[-1, 30, 55, 100],
        labels=["Green", "Amber", "Red"],
    ).astype(str)
    return out


def pmi_contributions(row: pd.Series) -> dict[str, float]:
    """Return the four PMI contributions for a single row (for the UI)."""
    return {
        "Load": round(float(row["pmi_load_contrib"]), 1),
        "Recovery": round(float(row["pmi_recovery_contrib"]), 1),
        "Neuromuscular": round(float(row["pmi_neuromuscular_contrib"]), 1),
        "Congestion": round(float(row["pmi_congestion_contrib"]), 1),
    }


if __name__ == "__main__":
    from src.data_processing import pipeline
    from src.feature_engineering.features import build_features

    out = pipeline.run_pipeline(persist=False)
    feats = build_features(out["daily"])
    mon = compute_pmi(feats)
    last = mon.sort_values("date").groupby("player_id").tail(1)
    print("PMI band distribution on last observed day per player:")
    print(last["pmi_band"].value_counts())
    print("\nDimensional status counts (full season):")
    for c in ["load_status", "recovery_status", "neuromuscular_status", "congestion_status"]:
        print(f"  {c}:", mon[c].value_counts().to_dict())
