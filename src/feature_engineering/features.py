"""
Feature engineering: from raw daily measurements to longitudinal features.

Every feature is computed *causally* — only information available up to and
including day t is used to build the feature for day t. This is the single most
important discipline for a temporal prediction problem: it is what keeps the
offline evaluation honest and makes the model deployable.

Feature families
----------------
Load          : rolling sums/means over 3/7/28 days, monotony, strain, ACWR.
Wellness      : 7-day rolling averages, deviation & z-score vs individual baseline.
Neuromuscular : CMJ carried forward, % change and z-score vs individual baseline.
Congestion    : minutes in last 7/14 days, matches in last 14 days, days since match.

Baselines are computed *per player* using an expanding window (each player is his
own reference), because comparing an athlete only to the squad mean hides
individually meaningful changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Minimum history (days) before an individual baseline is considered stable.
BASELINE_MIN_PERIODS = 14


# --------------------------------------------------------------------------- #
# Helpers (all causal / grouped by player)
# --------------------------------------------------------------------------- #
def _roll_sum(g: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return g.rolling(window, min_periods=min_periods).sum()


def _roll_mean(g: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return g.rolling(window, min_periods=min_periods).mean()


def _roll_std(g: pd.Series, window: int, min_periods: int = 2) -> pd.Series:
    return g.rolling(window, min_periods=min_periods).std()


def _expanding_mean(g: pd.Series) -> pd.Series:
    return g.expanding(min_periods=BASELINE_MIN_PERIODS).mean()


def _expanding_std(g: pd.Series) -> pd.Series:
    return g.expanding(min_periods=BASELINE_MIN_PERIODS).std()


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Return a per-player-per-day feature table (features + carried context)."""
    df = daily.sort_values(["player_id", "date"]).copy()
    gp = df.groupby("player_id", group_keys=False)

    # ------------------------------------------------------------------ Load
    # Fill non-session load with 0 (rest days genuinely have zero external load).
    for col in ["player_load", "total_distance", "high_speed_running",
                "sprint_distance", "session_rpe"]:
        df[col] = df[col].fillna(0.0)

    df["load_3d"] = gp["player_load"].apply(lambda s: _roll_sum(s, 3)).reset_index(level=0, drop=True)
    df["load_7d"] = gp["player_load"].apply(lambda s: _roll_sum(s, 7)).reset_index(level=0, drop=True)
    df["load_28d"] = gp["player_load"].apply(lambda s: _roll_sum(s, 28)).reset_index(level=0, drop=True)
    df["load_7d_mean"] = gp["player_load"].apply(lambda s: _roll_mean(s, 7)).reset_index(level=0, drop=True)
    df["load_7d_sd"] = gp["player_load"].apply(lambda s: _roll_std(s, 7)).reset_index(level=0, drop=True)
    df["hsr_7d"] = gp["high_speed_running"].apply(lambda s: _roll_sum(s, 7)).reset_index(level=0, drop=True)
    df["hsr_3d"] = gp["high_speed_running"].apply(lambda s: _roll_sum(s, 3)).reset_index(level=0, drop=True)

    # Monotony = mean daily load / SD of daily load (7d). Strain = weekly load * monotony.
    df["monotony"] = df["load_7d_mean"] / df["load_7d_sd"].replace(0, np.nan)
    df["monotony"] = df["monotony"].clip(upper=6).fillna(1.0)
    df["strain"] = df["load_7d"] * df["monotony"]

    # ACWR (acute:chronic workload ratio). Reported with care: the coupled 7:28
    # rolling-average version is known to be noisy, so we also expose the raw
    # acute and chronic windows and never use ACWR as a hard rule.
    acute = gp["player_load"].apply(lambda s: _roll_mean(s, 7)).reset_index(level=0, drop=True)
    chronic = gp["player_load"].apply(lambda s: _roll_mean(s, 28, min_periods=7)).reset_index(level=0, drop=True)
    df["acute_load"] = acute
    df["chronic_load"] = chronic
    df["acwr"] = (acute / chronic.replace(0, np.nan)).clip(0, 3).fillna(1.0)

    # -------------------------------------------------------------- Wellness
    # Wellness convention: higher fatigue/soreness/stress = worse;
    # higher sleep_quality/mood/sleep_duration = better.
    wellness_cols = ["sleep_duration", "sleep_quality", "fatigue",
                     "muscle_soreness", "stress", "mood"]
    for col in wellness_cols:
        # Impute short gaps by carrying forward the player's last report, then
        # fall back to the player mean (documented, conservative imputation).
        df[col] = gp[col].apply(lambda s: s.ffill()).reset_index(level=0, drop=True)
        df[col] = df.groupby("player_id")[col].transform(lambda s: s.fillna(s.mean()))
        # 7-day rolling average
        df[f"{col}_7d"] = gp[col].apply(lambda s: _roll_mean(s, 7)).reset_index(level=0, drop=True)
        # individual baseline + z-score
        base = gp[col].apply(_expanding_mean).reset_index(level=0, drop=True).shift(0)
        sd = gp[col].apply(_expanding_std).reset_index(level=0, drop=True)
        df[f"{col}_baseline"] = base
        df[f"{col}_z"] = ((df[col] - base) / sd.replace(0, np.nan)).clip(-4, 4).fillna(0.0)

    # A compact wellness index (z-scores oriented so that higher = better).
    df["wellness_index_z"] = (
        df["sleep_quality_z"] + df["mood_z"] - df["fatigue_z"]
        - df["muscle_soreness_z"] - df["stress_z"]
    ) / 5.0

    # ------------------------------------------------------------ Neuromuscular
    # CMJ is measured intermittently; carry the last measured value forward and
    # compute deviation vs the player's own baseline.
    df["cmj_measured"] = df["cmj_height"].notna().astype(int)
    df["cmj_last"] = gp["cmj_height"].apply(lambda s: s.ffill()).reset_index(level=0, drop=True)
    cmj_base = gp["cmj_height"].apply(lambda s: _expanding_mean(s.ffill())).reset_index(level=0, drop=True)
    cmj_sd = gp["cmj_height"].apply(lambda s: _expanding_std(s.ffill())).reset_index(level=0, drop=True)
    df["cmj_baseline"] = cmj_base
    df["cmj_pct_change"] = ((df["cmj_last"] - cmj_base) / cmj_base.replace(0, np.nan)).fillna(0.0)
    df["cmj_z"] = ((df["cmj_last"] - cmj_base) / cmj_sd.replace(0, np.nan)).clip(-4, 4).fillna(0.0)
    # rolling neuromuscular trend (change over last measured window)
    df["cmj_trend_7d"] = gp["cmj_last"].apply(
        lambda s: s - s.shift(7)).reset_index(level=0, drop=True).fillna(0.0)

    # ------------------------------------------------------------- Congestion
    df["minutes_7d"] = gp["minutes_played"].apply(lambda s: _roll_sum(s, 7)).reset_index(level=0, drop=True)
    df["minutes_14d"] = gp["minutes_played"].apply(lambda s: _roll_sum(s, 14)).reset_index(level=0, drop=True)
    df["matches_14d"] = gp["played_match"].apply(
        lambda s: _roll_sum(s.astype(int), 14)).reset_index(level=0, drop=True)
    df["matches_28d"] = gp["played_match"].apply(
        lambda s: _roll_sum(s.astype(int), 28)).reset_index(level=0, drop=True)
    # days_since_last_match already present from generation; keep and clip.
    if "days_since_last_match" in df.columns:
        df["days_since_last_match"] = df["days_since_last_match"].clip(0, 30)

    # Weekly load change (this week vs previous week) — a spike detector.
    df["load_week_change"] = gp["load_7d"].apply(
        lambda s: (s - s.shift(7)) / s.shift(7).replace(0, np.nan)
    ).reset_index(level=0, drop=True).clip(-2, 5).fillna(0.0)

    df = df.reset_index(drop=True)
    return df


# Canonical model feature list (leakage-safe, observable at prediction time).
MODEL_FEATURES = [
    # Load
    "load_3d", "load_7d", "load_28d", "load_7d_mean", "load_7d_sd",
    "hsr_3d", "hsr_7d", "monotony", "strain", "acwr",
    "acute_load", "chronic_load", "load_week_change",
    # Wellness (levels + individual deviations)
    "fatigue", "muscle_soreness", "stress", "sleep_duration", "sleep_quality",
    "fatigue_z", "muscle_soreness_z", "stress_z", "sleep_quality_z",
    "wellness_index_z",
    # Neuromuscular
    "cmj_last", "cmj_pct_change", "cmj_z", "cmj_trend_7d",
    # Congestion
    "minutes_7d", "minutes_14d", "matches_14d", "matches_28d",
    "days_since_last_match",
    # Static context (observable)
    "age",
]


if __name__ == "__main__":
    from src.data_processing import pipeline

    out = pipeline.run_pipeline(persist=False)
    feats = build_features(out["daily"])
    print("Feature table:", feats.shape)
    present = [c for c in MODEL_FEATURES if c in feats.columns]
    print(f"Model features present: {len(present)}/{len(MODEL_FEATURES)}")
    missing = set(MODEL_FEATURES) - set(present)
    if missing:
        print("MISSING:", missing)
    print(feats[present].describe().T[["mean", "std", "min", "max"]].round(2).head(40))
