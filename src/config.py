"""
Central configuration for Football Performance Intelligence.

Everything that is likely to change (paths, the random seed, thresholds used by
the monitoring rules, the ML target definition) lives here so that the rest of
the codebase never hard-codes a magic number. This is also the seam that makes
the project portable: swap the SQLite URI for a PostgreSQL one and nothing else
in the data layer needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"

DB_PATH = DATA_DIR / "fpi.db"
# Data layer is written against SQLAlchemy-style URIs so the store can later be
# swapped for PostgreSQL without touching call sites.
DB_URI = f"sqlite:///{DB_PATH}"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Season definition
# --------------------------------------------------------------------------- #
SEASON_START = "2024-07-15"   # pre-season start
SEASON_END = "2025-05-25"     # last match week
N_PLAYERS = 28

POSITIONS = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]
# Rough squad composition (adds up to ~28)
POSITION_COUNTS = {
    "GK": 3, "CB": 5, "FB": 4, "DM": 3, "CM": 4, "AM": 3, "W": 3, "ST": 3,
}

# --------------------------------------------------------------------------- #
# ML target definition
# --------------------------------------------------------------------------- #
# We predict, on a given monitoring day, whether the player will enter a
# "reduced availability" state (modified training OR unavailable) within the
# next PREDICTION_HORIZON_DAYS. This is an operational early-warning task, not a
# medical diagnosis.
PREDICTION_HORIZON_DAYS = 7

# Temporal split boundaries (fraction of the season timeline).
TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15   # test is the remaining 0.15


@dataclass(frozen=True)
class MonitoringThresholds:
    """Thresholds used by the rule-based dimensional monitoring.

    These encode widely-used sports-science heuristics (ACWR sweet spot,
    z-score deviations from an individual baseline). They are deliberately
    conservative and fully documented; they are decision-support cues, not
    hard rules.
    """
    acwr_low: float = 0.8
    acwr_high: float = 1.3        # classic "sweet spot" upper bound
    acwr_very_high: float = 1.5
    monotony_high: float = 2.0
    # Wellness / neuromuscular deviations expressed as individual z-scores.
    z_moderate: float = -1.0
    z_poor: float = -1.5
    cmj_drop_reduced: float = -0.10   # 10% below individual baseline
    cmj_drop_critical: float = -0.15  # 15% below individual baseline
    congestion_minutes_7d_high: int = 200
    congestion_matches_14d_high: int = 3


THRESHOLDS = MonitoringThresholds()

# Weighting of the Performance Monitoring Index (must sum to 1.0).
# Documented in the README methodology section.
PMI_WEIGHTS = {
    "load": 0.25,
    "recovery": 0.25,
    "neuromuscular": 0.30,
    "congestion": 0.20,
}

# Feature columns that must NEVER enter the model because they are only known
# at or after the target event (data-leakage guard, enforced in ml/dataset.py).
LEAKAGE_BLOCKLIST = [
    "available", "modified_training", "unavailable", "availability_status",
    "target", "target_event_date", "days_to_event",
]
