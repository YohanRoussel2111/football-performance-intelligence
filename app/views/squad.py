"""Squad Overview — availability counts, squad load, and the full monitoring
table (Player | Position | Availability | Load | Recovery | CMJ | ML)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import app_core, ui
from src.visualization import charts


def render():
    ui.header("Squad Overview", "full-squad monitoring snapshot")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    snap = app_core.latest_snapshot(a["table"], scores, a["players"])

    # ---- KPI row ----
    avail = (snap["availability_status"] == "available").sum()
    monitor = (snap["monitoring_level"] == "HIGH").sum()
    modified = (snap["availability_status"] == "modified_training").sum()
    unavail = (snap["availability_status"] == "unavailable").sum()
    squad_load = snap["load_7d"].mean()
    poor_recov = (snap["recovery_status"] == "Poor").sum()

    c = st.columns(6)
    c[0].metric("Available", int(avail))
    c[1].metric("HIGH monitoring", int(monitor))
    c[2].metric("Modified", int(modified))
    c[3].metric("Unavailable", int(unavail))
    c[4].metric("Mean 7d load", f"{squad_load:,.0f}")
    c[5].metric("Poor recovery", int(poor_recov))

    next_match = _next_match(a)
    if next_match is not None:
        st.caption(f"🗓️ Upcoming fixture: **{next_match['opponent']}** "
                   f"({next_match['competition']}) — {pd.to_datetime(next_match['date']).date()}")

    st.divider()
    left, right = st.columns([1.55, 1])

    with left:
        st.subheader("Monitoring table")
        table = _build_table(snap)
        st.dataframe(
            table, width='stretch', hide_index=True, height=560,
            column_config={
                "Risk": st.column_config.ProgressColumn(
                    "ML risk", format="%.0f%%", min_value=0, max_value=100,
                    help="Calibrated probability of a reduced-availability event "
                         "in the next 7 days."),
                "Load 7d": st.column_config.NumberColumn(format="%.0f"),
                "ACWR": st.column_config.NumberColumn(format="%.2f"),
                "CMJ Δ%": st.column_config.NumberColumn(
                    format="%.1f%%", help="CMJ vs the player's own baseline."),
            },
        )

    with right:
        st.subheader("Squad 7-day load")
        st.plotly_chart(charts.squad_load_bar(snap), width='stretch')

    ui.disclaimer()


def _next_match(a):
    matches = a["matches"].copy()
    matches["date"] = pd.to_datetime(matches["date"])
    last = pd.to_datetime(a["table"]["date"]).max()
    upcoming = matches[matches["date"] > last].sort_values("date")
    if len(upcoming):
        return upcoming.iloc[0]
    # demo season ends on a match; show the final fixture instead
    return matches.sort_values("date").iloc[-1]


def _build_table(snap: pd.DataFrame) -> pd.DataFrame:
    df = snap.copy()
    df["CMJ Δ%"] = df["cmj_pct_change"] * 100
    out = pd.DataFrame({
        "Player": df["player_name"],
        "Pos": df["position"],
        "Availability": df["availability_status"].map(
            {"available": "🟢 Available", "modified_training": "🟡 Modified",
             "unavailable": "🔴 Unavailable"}),
        "Load": df["load_status"].map({"Normal": "🟢 Normal", "Elevated": "🟡 Elevated",
                                       "High": "🔴 High"}),
        "Recovery": df["recovery_status"].map({"Good": "🟢 Good", "Moderate": "🟡 Moderate",
                                               "Poor": "🔴 Poor"}),
        "Neuro": df["neuromuscular_status"].map({"Stable": "🟢 Stable", "Reduced": "🟡 Reduced",
                                                 "Critical": "🔴 Critical"}),
        "Load 7d": df["load_7d"],
        "ACWR": df["acwr"],
        "CMJ Δ%": df["CMJ Δ%"],
        "ML level": df["monitoring_level"].map({"LOW": "🟢 LOW", "MODERATE": "🟡 MODERATE",
                                                "HIGH": "🔴 HIGH"}),
        "Risk": (df["risk_probability"] * 100),
    })
    order = {"🔴 HIGH": 0, "🟡 MODERATE": 1, "🟢 LOW": 2}
    out = out.sort_values(["ML level", "Risk"],
                          key=lambda s: s.map(order) if s.name == "ML level" else -s)
    return out
