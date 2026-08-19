"""
Data engineering pipeline.

Responsibilities:
    1. Ingest raw CSVs (or the generated season).
    2. Validate schema & types.
    3. Detect missing values, duplicates and outliers.
    4. Harmonize units and de-duplicate.
    5. Build daily and weekly aggregates.
    6. Persist clean tables to SQLite.
    7. Emit a machine-readable Data Quality report.

Design note: cleaning is *non-destructive and logged*. Outliers are flagged and
capped, not silently dropped, and every action is captured in the quality report
so a practitioner can see exactly what the pipeline did to their data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd

from src import config
from src.data_processing import database as db
from src.data_processing import generate_synthetic

# Physiological plausibility bounds used for outlier detection / capping.
# (min, max) — values outside are flagged; hard-impossible values are capped.
PLAUSIBLE_BOUNDS = {
    "total_distance": (0, 16000),
    "high_speed_running": (0, 1600),
    "sprint_distance": (0, 700),
    "player_load": (0, 1300),
    "session_rpe": (0, 1200),
    "sleep_duration": (3, 12),
    "cmj_height": (15, 55),
    "sprint_10m": (1.4, 2.2),
    "sprint_30m": (3.4, 5.0),
    "heart_rate_mean": (35, 210),
}

REQUIRED_TABLES = ["players", "gps_data", "wellness", "availability"]


@dataclass
class QualityReport:
    generated_at: str
    n_players: int
    n_observations: int
    date_min: str
    date_max: str
    last_update: str
    freshness_days: int
    duplicate_records: int
    missing_by_column: dict
    completeness_pct: dict
    outliers_by_column: dict
    outliers_capped: int
    overall_completeness: float
    quality_flag: str  # OK / WARNING / INSUFFICIENT

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def ingest_from_generator(seed: int = config.RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """Generate a season and return raw tables (pre-cleaning)."""
    return generate_synthetic.generate_season(seed=seed)


def ingest_from_csv(directory) -> dict[str, pd.DataFrame]:
    """Load raw CSVs named like the canonical tables from a directory."""
    from pathlib import Path

    directory = Path(directory)
    tables = {}
    for name in db.TABLES:
        f = directory / f"{name}.csv"
        if f.exists():
            df = pd.read_csv(f)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            tables[name] = df
    return tables


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_schema(tables: dict[str, pd.DataFrame]) -> list[str]:
    problems = []
    for req in REQUIRED_TABLES:
        if req not in tables or tables[req].empty:
            problems.append(f"Missing or empty required table: {req}")
    if "availability" in tables:
        av = tables["availability"]
        bad = set(av["availability_status"].dropna().unique()) - {
            "available", "modified_training", "unavailable"}
        if bad:
            problems.append(f"Unexpected availability labels: {bad}")
    return problems


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _harmonize_units(daily: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent units. Here distances are metres, times seconds.

    In a real pipeline this is where km->m, min->s conversions and vendor-
    specific column mapping would live. We assert the ranges are sane.
    """
    daily = daily.copy()
    # Example harmonization: if total_distance looks like km (small numbers on
    # non-rest days), convert to metres.
    med = daily.loc[daily["total_distance"] > 0, "total_distance"].median()
    if med < 100:  # clearly kilometres
        daily["total_distance"] *= 1000
    return daily


def clean_daily(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """De-duplicate, harmonize units, flag & cap outliers.

    Returns cleaned daily frame and a stats dict for the quality report.
    """
    stats = {}
    n0 = len(daily)

    # 1. duplicates (same player + date, identical rows from a double import)
    dupe_mask = daily.duplicated(subset=["player_id", "date"], keep="first")
    stats["duplicate_records"] = int(dupe_mask.sum())
    daily = daily.loc[~dupe_mask].copy()

    # 2. harmonize units
    daily = _harmonize_units(daily)

    # 3. outlier detection + capping against plausibility bounds
    outliers = {}
    capped = 0
    for col, (lo, hi) in PLAUSIBLE_BOUNDS.items():
        if col not in daily.columns:
            continue
        s = daily[col]
        mask = s.notna() & ((s < lo) | (s > hi))
        cnt = int(mask.sum())
        if cnt:
            outliers[col] = cnt
            # cap to bounds (non-destructive: value corrected, row retained)
            daily.loc[daily[col].notna() & (daily[col] < lo), col] = lo
            daily.loc[daily[col].notna() & (daily[col] > hi), col] = hi
            capped += cnt
    stats["outliers_by_column"] = outliers
    stats["outliers_capped"] = capped
    stats["rows_removed_duplicates"] = n0 - len(daily)
    return daily.reset_index(drop=True), stats


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def build_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned per-player-per-day table (already daily-grained)."""
    return daily.sort_values(["player_id", "date"]).reset_index(drop=True)


def build_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """ISO-week aggregates per player — the granularity most staff review."""
    d = daily.copy()
    d["iso_year"] = d["date"].dt.isocalendar().year
    d["iso_week"] = d["date"].dt.isocalendar().week
    agg = (
        d.groupby(["player_id", "iso_year", "iso_week"])
        .agg(
            week_start=("date", "min"),
            total_distance=("total_distance", "sum"),
            high_speed_running=("high_speed_running", "sum"),
            sprint_distance=("sprint_distance", "sum"),
            player_load=("player_load", "sum"),
            session_rpe=("session_rpe", "sum"),
            minutes_played=("minutes_played", "sum"),
            training_days=("day_type", lambda s: int((s == "training").sum())),
            matches=("played_match", "sum"),
            mean_fatigue=("fatigue", "mean"),
            mean_soreness=("muscle_soreness", "mean"),
            mean_sleep=("sleep_duration", "mean"),
            days_unavailable=("unavailable", "sum"),
            days_modified=("modified_training", "sum"),
        )
        .reset_index()
    )
    return agg


# --------------------------------------------------------------------------- #
# Quality report
# --------------------------------------------------------------------------- #
_QUALITY_COLUMNS = [
    "total_distance", "high_speed_running", "player_load", "session_rpe",
    "sleep_duration", "sleep_quality", "fatigue", "muscle_soreness",
    "stress", "mood",
]


def build_quality_report(daily: pd.DataFrame, clean_stats: dict) -> QualityReport:
    n_obs = len(daily)
    missing = {}
    completeness = {}
    for col in _QUALITY_COLUMNS:
        if col in daily.columns:
            m = int(daily[col].isna().sum())
            missing[col] = m
            completeness[col] = round(100 * (1 - m / max(n_obs, 1)), 2)
    overall = round(float(np.mean(list(completeness.values()))), 2) if completeness else 0.0

    date_max = daily["date"].max()
    # "Freshness" measured against the season end (the demo's 'today').
    reference_today = pd.Timestamp(config.SEASON_END)
    freshness = int((reference_today - date_max).days)

    if overall < 85 or clean_stats.get("duplicate_records", 0) > 50:
        flag = "INSUFFICIENT"
    elif overall < 95 or freshness > 3:
        flag = "WARNING"
    else:
        flag = "OK"

    return QualityReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        n_players=daily["player_id"].nunique(),
        n_observations=n_obs,
        date_min=str(daily["date"].min().date()),
        date_max=str(date_max.date()),
        last_update=str(date_max.date()),
        freshness_days=freshness,
        duplicate_records=clean_stats.get("duplicate_records", 0),
        missing_by_column=missing,
        completeness_pct=completeness,
        outliers_by_column=clean_stats.get("outliers_by_column", {}),
        outliers_capped=clean_stats.get("outliers_capped", 0),
        overall_completeness=overall,
        quality_flag=flag,
    )


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(seed: int = config.RANDOM_SEED, persist: bool = True) -> dict:
    """End-to-end: generate -> validate -> clean -> aggregate -> persist."""
    raw = ingest_from_generator(seed=seed)
    problems = validate_schema(raw)

    daily_raw = raw["daily"]
    daily_clean, clean_stats = clean_daily(daily_raw)
    daily = build_daily(daily_clean)
    weekly = build_weekly(daily)
    report = build_quality_report(daily, clean_stats)

    if persist:
        db.init_schema()
        # Rebuild normalized tables from the cleaned daily frame so the DB and
        # the analytics layer are always consistent.
        db.write_table("players", raw["players"])
        db.write_table("matches", raw["matches"])
        db.write_table("sessions", raw["sessions"])
        db.write_table("gps_data", daily[["player_id", "date"] + [
            c for c in raw["gps_data"].columns if c not in ("player_id", "date")]])
        db.write_table("wellness", daily[["player_id", "date", "sleep_duration",
            "sleep_quality", "fatigue", "muscle_soreness", "stress", "mood"]])
        db.write_table("physical_tests", raw["physical_tests"])
        db.write_table("availability", daily[["player_id", "date",
            "availability_status", "available", "modified_training",
            "unavailable", "minutes_played", "days_since_last_match"]])
        db.write_table("availability_episodes", raw["availability_episodes"])

    return {
        "raw": raw,
        "daily": daily,
        "weekly": weekly,
        "quality_report": report,
        "validation_problems": problems,
        "clean_stats": clean_stats,
    }


if __name__ == "__main__":
    out = run_pipeline()
    r = out["quality_report"]
    print("Validation problems:", out["validation_problems"])
    print(f"Observations: {r.n_observations}  Players: {r.n_players}")
    print(f"Duplicates removed: {r.duplicate_records}  Outliers capped: {r.outliers_capped}")
    print(f"Overall completeness: {r.overall_completeness}%  Flag: {r.quality_flag}")
    print(f"Weekly rows: {len(out['weekly'])}")
