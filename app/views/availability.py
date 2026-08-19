"""Disponibilité — chronologie de disponibilité de l'effectif et historique des épisodes."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import app_core, ui
from src.visualization import charts


def render():
    ui.header("Disponibilité", "disponibilité de l'effectif sur la saison")
    a = app_core.get_analysis()
    table = a["table"]
    players = a["players"]
    episodes = a["episodes"]

    total_player_days = len(table)
    unavailable_days = int((table["availability_status"] == "unavailable").sum())
    modified_days = int((table["availability_status"] == "modified_training").sum())
    n_eps = len(episodes) if episodes is not None else 0
    avail_pct = 100 * (1 - (unavailable_days + modified_days) / total_player_days)

    c = st.columns(4)
    c[0].metric("Disponibilité (saison)", f"{avail_pct:.1f}%")
    c[1].metric("Jours-joueur indisponibles", unavailable_days)
    c[2].metric("Jours-joueur aménagés", modified_days)
    c[3].metric("Épisodes au total", n_eps)

    st.divider()
    st.subheader("Chronologie de disponibilité")
    st.caption("Chaque ligne est un joueur ; rouge = indisponible, orange = "
               "entraînement aménagé.")
    st.plotly_chart(_timeline(table, players), width='stretch')

    st.subheader("Journal des épisodes")
    if episodes is not None and len(episodes):
        eps = episodes.merge(players[["player_id", "player_name", "position"]],
                             on="player_id", how="left").sort_values("start_date")
        show = pd.DataFrame({
            "Joueur": eps["player_name"], "Poste": eps["position"],
            "Type": eps["type"].map({"modified": "🟡 Aménagé", "unavailable": "🔴 Indisponible"}),
            "Début": pd.to_datetime(eps["start_date"]).dt.date,
            "Fin": pd.to_datetime(eps["end_date"]).dt.date,
            "Jours": eps["duration_days"],
        })
        st.dataframe(show, hide_index=True, width='stretch', height=360)
    else:
        st.info("Aucun épisode enregistré.")
    ui.disclaimer()


def _timeline(table, players):
    d = table.copy()
    d["date"] = pd.to_datetime(d["date"])
    name_map = players.set_index("player_id")["player_name"].to_dict()
    pid_order = list(players.sort_values("player_id")["player_id"])
    code = {"available": 0, "modified_training": 1, "unavailable": 2}
    piv = d.pivot_table(index="player_id", columns="date",
                        values="availability_status", aggfunc="first")
    piv = piv.reindex(index=pid_order)
    z = piv.map(lambda s: code.get(s, 0)).values
    fig = go.Figure(go.Heatmap(
        z=z, x=piv.columns, y=[name_map.get(i, i) for i in piv.index],
        colorscale=[[0, charts.GOOD], [0.5, charts.WARN], [1, charts.BAD]],
        zmin=0, zmax=2, showscale=False,
        hovertemplate="%{y}<br>%{x|%d %b}<extra></extra>"))
    return charts.apply_theme(fig, height=max(460, 15 * len(piv)))
