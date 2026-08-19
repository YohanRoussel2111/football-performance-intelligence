"""Suivi de la charge — charge externe/interne, ACWR et congestion de l'effectif."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import app_core, ui
from src.visualization import charts


def render():
    ui.header("Suivi de la charge", "charge externe & interne de l'effectif")
    a = app_core.get_analysis()
    table = a["table"]
    players = a["players"]

    ui.science_box(
        "La <b>monotonie</b> et la <b>contrainte (strain)</b> de Foster (2001), et le "
        "<b>ratio charge aiguë:chronique (ACWR)</b> de Gabbett (2016), sont des "
        "indicateurs classiques de gestion de la charge. La « zone optimale » "
        "(ACWR ~0,8–1,3) est un repère, pas une règle absolue (Impellizzeri, 2020).")

    win = st.select_slider("Fenêtre glissante (jours)", options=[14, 28, 42, 60, 90],
                           value=42)
    last = pd.to_datetime(table["date"]).max()
    recent = table[pd.to_datetime(table["date"]) > last - pd.Timedelta(days=win)]

    # `table` contient déjà player_name & position.
    snap = table.sort_values("date").groupby("player_id").tail(1)

    c = st.columns(4)
    c[0].metric("Charge 7 j moyenne", f"{snap['load_7d'].mean():,.0f}")
    c[1].metric("ACWR moyen", f"{snap['acwr'].mean():.2f}")
    c[2].metric("Joueurs ACWR > 1,3", int((snap["acwr"] > 1.3).sum()))
    c[3].metric("Congestion élevée", int((snap["congestion_status"] == "High").sum()))

    st.divider()
    st.subheader("Carte de chaleur de la charge (player load)")
    st.plotly_chart(_heatmap(recent, players), width='stretch')

    left, right = st.columns(2)
    with left:
        st.subheader("Distribution de l'ACWR")
        st.plotly_chart(_acwr_hist(snap), width='stretch')
    with right:
        st.subheader("Congestion (minutes sur 14 jours)")
        cong = snap.sort_values("minutes_14d", ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            x=cong["minutes_14d"], y=cong["player_name"], orientation="h",
            marker_color=charts.PRIMARY,
            hovertemplate="%{y}<br>%{x:.0f} min / 14 j<extra></extra>"))
        st.plotly_chart(charts.apply_theme(fig, height=420), width='stretch')

    ui.disclaimer()


def _heatmap(recent, players):
    d = recent.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week"] = d["date"].dt.strftime("%d %b")
    piv = d.pivot_table(index="player_id", columns="week", values="player_load",
                        aggfunc="mean")
    order = (d.groupby("week")["date"].min().sort_values().index)
    piv = piv.reindex(columns=order)
    name_map = players.set_index("player_id")["player_name"].to_dict()
    piv.index = [name_map.get(i, i) for i in piv.index]
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index, colorscale="Teal",
        hovertemplate="%{y}<br>%{x}<br>charge : %{z:.0f}<extra></extra>",
        colorbar=dict(title="charge")))
    return charts.apply_theme(fig, height=max(420, 15 * len(piv)))


def _acwr_hist(snap):
    fig = go.Figure(go.Histogram(x=snap["acwr"], nbinsx=18, marker_color=charts.PRIMARY))
    fig.add_vrect(x0=0.8, x1=1.3, fillcolor=charts.GOOD, opacity=0.12, line_width=0)
    fig.add_vline(x=1.5, line=dict(color=charts.BAD, dash="dash"))
    fig.update_xaxes(title="ACWR")
    return charts.apply_theme(fig, height=420)
