"""Availability — squad availability timeline and episode history."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import app_core, ui
from src.visualization import charts


def render():
    ui.header("Availability", "squad availability over the season")
    a = app_core.get_analysis()
    table = a["table"]
    players = a["players"]
    episodes = a["episodes"]

    # season KPIs
    total_player_days = len(table)
    unavailable_days = int((table["availability_status"] == "unavailable").sum())
    modified_days = int((table["availability_status"] == "modified_training").sum())
    n_eps = len(episodes) if episodes is not None else 0
    avail_pct = 100 * (1 - (unavailable_days + modified_days) / total_player_days)

    c = st.columns(4)
    c[0].metric("Season availability", f"{avail_pct:.1f}%")
    c[1].metric("Unavailable player-days", unavailable_days)
    c[2].metric("Modified player-days", modified_days)
    c[3].metric("Total episodes", n_eps)

    st.divider()
    st.subheader("Availability timeline")
    st.caption("Each row is a player; red = unavailable, amber = modified training.")
    st.plotly_chart(_timeline(table, players), width='stretch')

    st.subheader("Episode log")
    if episodes is not None and len(episodes):
        eps = episodes.merge(players[["player_id", "player_name", "position"]],
                             on="player_id", how="left").sort_values("start_date")
        show = pd.DataFrame({
            "Player": eps["player_name"], "Pos": eps["position"],
            "Type": eps["type"].map({"modified": "🟡 Modified", "unavailable": "🔴 Unavailable"}),
            "Start": pd.to_datetime(eps["start_date"]).dt.date,
            "End": pd.to_datetime(eps["end_date"]).dt.date,
            "Days": eps["duration_days"],
        })
        st.dataframe(show, hide_index=True, width='stretch', height=360)
    else:
        st.info("No episodes recorded.")
    ui.disclaimer()


def _timeline(table, players):
    d = table.copy()
    d["date"] = pd.to_datetime(d["date"])
    name_map = players.set_index("player_id")["player_name"].to_dict()
    pid_order = list(players.sort_values("player_id")["player_id"])
    code = {"available": 0, "modified_training": 1, "unavailable": 2}
    piv = d.pivot_table(index="player_id", columns="date",
                        values="availability_status",
                        aggfunc="first")
    piv = piv.reindex(index=pid_order)
    z = piv.map(lambda s: code.get(s, 0)).values  # DataFrame.map (pandas ≥2.1)
    fig = go.Figure(go.Heatmap(
        z=z, x=piv.columns, y=[name_map.get(i, i) for i in piv.index],
        colorscale=[[0, charts.GOOD], [0.5, charts.WARN], [1, charts.BAD]],
        zmin=0, zmax=2, showscale=False,
        hovertemplate="%{y}<br>%{x|%d %b}<extra></extra>"))
    return charts.apply_theme(fig, height=max(460, 15 * len(piv)))
