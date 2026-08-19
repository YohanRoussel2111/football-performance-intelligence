"""
SQLite persistence layer.

The store is deliberately thin and URI-driven. Every function takes an optional
connection URI so the same code can target SQLite today and PostgreSQL later:
only `get_engine` would need to learn a new dialect. We use the stdlib `sqlite3`
module (no SQLAlchemy dependency required to run the demo), but keep the schema
and access patterns portable.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src import config

# Canonical table list exposed to the rest of the app.
TABLES = [
    "players", "sessions", "matches", "gps_data", "wellness",
    "physical_tests", "availability", "availability_episodes",
    "features", "predictions",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id     TEXT PRIMARY KEY,
    player_name   TEXT NOT NULL,
    age           INTEGER,
    position      TEXT,
    dominant_leg  TEXT,
    height_cm     INTEGER,
    weight_kg     INTEGER
);

CREATE TABLE IF NOT EXISTS matches (
    match_id     TEXT PRIMARY KEY,
    date         TEXT,
    md_code      TEXT,
    competition  TEXT,
    opponent     TEXT,
    is_home      INTEGER,
    congested    INTEGER,
    preseason    INTEGER
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id    INTEGER PRIMARY KEY,
    player_id     TEXT REFERENCES players(player_id),
    date          TEXT,
    day_type      TEXT,
    md_code       TEXT,
    session_rpe   REAL,
    minutes_played INTEGER,
    played_match  INTEGER
);

CREATE TABLE IF NOT EXISTS gps_data (
    player_id          TEXT REFERENCES players(player_id),
    date               TEXT,
    total_distance     REAL,
    high_speed_running REAL,
    sprint_distance    REAL,
    accelerations      INTEGER,
    decelerations      INTEGER,
    player_load        REAL,
    metabolic_load     REAL,
    PRIMARY KEY (player_id, date)
);

CREATE TABLE IF NOT EXISTS wellness (
    player_id      TEXT REFERENCES players(player_id),
    date           TEXT,
    sleep_duration REAL,
    sleep_quality  REAL,
    fatigue        REAL,
    muscle_soreness REAL,
    stress         REAL,
    mood           REAL,
    PRIMARY KEY (player_id, date)
);

CREATE TABLE IF NOT EXISTS physical_tests (
    player_id            TEXT REFERENCES players(player_id),
    date                 TEXT,
    cmj_height           REAL,
    cmj_contraction_time REAL,
    eccentric_force      REAL,
    concentric_force     REAL,
    peak_force           REAL,
    sprint_10m           REAL,
    sprint_30m           REAL,
    PRIMARY KEY (player_id, date)
);

CREATE TABLE IF NOT EXISTS availability (
    player_id           TEXT REFERENCES players(player_id),
    date                TEXT,
    availability_status TEXT,
    available           INTEGER,
    modified_training   INTEGER,
    unavailable         INTEGER,
    minutes_played      INTEGER,
    days_since_last_match INTEGER,
    PRIMARY KEY (player_id, date)
);

CREATE TABLE IF NOT EXISTS availability_episodes (
    player_id     TEXT REFERENCES players(player_id),
    start_date    TEXT,
    end_date      TEXT,
    type          TEXT,
    duration_days INTEGER
);

CREATE TABLE IF NOT EXISTS features (
    player_id TEXT,
    date      TEXT,
    PRIMARY KEY (player_id, date)
);

CREATE TABLE IF NOT EXISTS predictions (
    player_id      TEXT,
    date           TEXT,
    model_version  TEXT,
    risk_probability REAL,
    monitoring_level TEXT,
    generated_at   TEXT,
    PRIMARY KEY (player_id, date, model_version)
);

CREATE INDEX IF NOT EXISTS idx_gps_player ON gps_data(player_id, date);
CREATE INDEX IF NOT EXISTS idx_avail_player ON availability(player_id, date);
"""


def _db_file(uri: str | None = None) -> Path:
    uri = uri or config.DB_URI
    if uri.startswith("sqlite:///"):
        return Path(uri.replace("sqlite:///", ""))
    raise NotImplementedError(
        f"Only sqlite URIs are wired up in this MVP; got {uri!r}. "
        "The access layer is URI-driven so a PostgreSQL engine can slot in here."
    )


def get_connection(uri: str | None = None) -> sqlite3.Connection:
    path = _db_file(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_schema(uri: str | None = None) -> None:
    with get_connection(uri) as conn:
        conn.executescript(SCHEMA)


def write_table(name: str, df: pd.DataFrame, uri: str | None = None,
                if_exists: str = "replace") -> None:
    df = df.copy()
    # Normalize datetimes to ISO strings for portable storage.
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    with get_connection(uri) as conn:
        df.to_sql(name, conn, if_exists=if_exists, index=False)


def read_table(name: str, uri: str | None = None) -> pd.DataFrame:
    with get_connection(uri) as conn:
        df = pd.read_sql(f"SELECT * FROM {name}", conn)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def run_query(sql: str, uri: str | None = None) -> pd.DataFrame:
    with get_connection(uri) as conn:
        return pd.read_sql(sql, conn)


def table_exists(name: str, uri: str | None = None) -> bool:
    with get_connection(uri) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
        return cur.fetchone() is not None
