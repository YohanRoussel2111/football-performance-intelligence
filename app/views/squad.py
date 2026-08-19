"""Vue Effectif — compteurs de disponibilité, charge de l'effectif, et tableau de
suivi complet (Joueur | Poste | Disponibilité | Charge | Récupération | CMJ | ML)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import app_core, ui
from src.visualization import charts


def render():
    ui.header("Vue d'ensemble de l'effectif", "instantané de suivi de tout le groupe")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    snap = app_core.latest_snapshot(a["table"], scores, a["players"])

    # ---- ligne d'indicateurs ----
    avail = (snap["availability_status"] == "available").sum()
    monitor = (snap["monitoring_level"] == "HIGH").sum()
    modified = (snap["availability_status"] == "modified_training").sum()
    unavail = (snap["availability_status"] == "unavailable").sum()
    squad_load = snap["load_7d"].mean()
    poor_recov = (snap["recovery_status"] == "Poor").sum()

    c = st.columns(6)
    c[0].metric("Disponibles", int(avail))
    c[1].metric("Surveillance ÉLEVÉE", int(monitor))
    c[2].metric("Aménagés", int(modified))
    c[3].metric("Indisponibles", int(unavail))
    c[4].metric("Charge 7 j moyenne", f"{squad_load:,.0f}")
    c[5].metric("Récupération faible", int(poor_recov))

    next_match = _next_match(a)
    if next_match is not None:
        st.caption(f"🗓️ Prochaine rencontre : **{next_match['opponent']}** "
                   f"({next_match['competition']}) — {pd.to_datetime(next_match['date']).date()}")

    st.divider()
    left, right = st.columns([1.55, 1])

    with left:
        st.subheader("Tableau de suivi")
        table = _build_table(snap)
        st.dataframe(
            table, width='stretch', hide_index=True, height=560,
            column_config={
                "Risque": st.column_config.ProgressColumn(
                    "Risque ML", format="%.0f%%", min_value=0, max_value=100,
                    help="Probabilité calibrée d'un épisode de disponibilité réduite "
                         "dans les 7 prochains jours."),
                "Charge 7j": st.column_config.NumberColumn(format="%.0f"),
                "ACWR": st.column_config.NumberColumn(format="%.2f"),
                "CMJ Δ%": st.column_config.NumberColumn(
                    format="%.1f%%", help="CMJ vs la référence propre au joueur."),
            },
        )

    with right:
        st.subheader("Charge de l'effectif (7 j)")
        st.plotly_chart(charts.squad_load_bar(snap), width='stretch')

    ui.disclaimer()


def _next_match(a):
    matches = a["matches"].copy()
    matches["date"] = pd.to_datetime(matches["date"])
    last = pd.to_datetime(a["table"]["date"]).max()
    upcoming = matches[matches["date"] > last].sort_values("date")
    if len(upcoming):
        return upcoming.iloc[0]
    return matches.sort_values("date").iloc[-1]


def _build_table(snap: pd.DataFrame) -> pd.DataFrame:
    df = snap.copy()
    df["CMJ Δ%"] = df["cmj_pct_change"] * 100
    out = pd.DataFrame({
        "Joueur": df["player_name"],
        "Poste": df["position"],
        "Disponibilité": df["availability_status"].map(ui.fr_avail_tag),
        "Charge": df["load_status"].map(ui.fr_load_tag),
        "Récup.": df["recovery_status"].map(ui.fr_recov_tag),
        "Neuro": df["neuromuscular_status"].map(ui.fr_nm_tag),
        "Charge 7j": df["load_7d"],
        "ACWR": df["acwr"],
        "CMJ Δ%": df["CMJ Δ%"],
        "Niveau ML": df["monitoring_level"].map(ui.fr_level_tag),
        "Risque": (df["risk_probability"] * 100),
    })
    order = {"🔴 ÉLEVÉ": 0, "🟡 MODÉRÉ": 1, "🟢 FAIBLE": 2}
    out = out.sort_values(["Niveau ML", "Risque"],
                          key=lambda s: s.map(order) if s.name == "Niveau ML" else -s)
    return out
