"""Qualité des données — complétude, valeurs manquantes, doublons, aberrations, fraîcheur."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import app_core, ui
from src.visualization import charts


def render():
    ui.header("Qualité des données", "faire confiance aux entrées avant le modèle")
    a = app_core.get_analysis()
    qr = a["quality_report"]

    flag = qr.quality_flag
    color = {"OK": charts.GOOD, "WARNING": charts.WARN, "INSUFFICIENT": charts.BAD}[flag]
    st.markdown(
        f"<div class='card'><span class='metric-label'>Qualité globale des données</span><br>"
        f"<span style='color:{color};font-size:1.4rem;font-weight:700'>"
        f"{ui.FR_QUALITY.get(flag, flag)}</span>"
        f"  ·  complétude {qr.overall_completeness:.1f}%</div>",
        unsafe_allow_html=True)
    if flag == "INSUFFICIENT":
        st.error("Qualité des données insuffisante — les sorties du modèle ne "
                 "devraient pas être utilisées tant que les entrées ne sont pas corrigées.")
    elif flag == "WARNING":
        st.warning("Avertissements sur la qualité des données — interpréter les "
                   "sorties du modèle avec prudence.")

    st.divider()
    c = st.columns(6)
    c[0].metric("Observations", f"{qr.n_observations:,}")
    c[1].metric("Joueurs", qr.n_players)
    c[2].metric("Doublons", qr.duplicate_records)
    c[3].metric("Aberrations bornées", qr.outliers_capped)
    c[4].metric("Dernière mise à jour", qr.last_update)
    c[5].metric("Fraîcheur", f"{qr.freshness_days} j")

    st.caption(f"Couverture : {qr.date_min} → {qr.date_max}")

    left, right = st.columns(2)
    with left:
        st.subheader("Complétude par champ")
        comp = pd.DataFrame({
            "field": list(qr.completeness_pct.keys()),
            "completeness": list(qr.completeness_pct.values()),
        }).sort_values("completeness")
        fig = go.Figure(go.Bar(
            x=comp["completeness"], y=comp["field"], orientation="h",
            marker_color=[charts.GOOD if v >= 95 else (charts.WARN if v >= 90 else charts.BAD)
                          for v in comp["completeness"]],
            hovertemplate="%{y} : %{x:.1f}%<extra></extra>"))
        fig.update_xaxes(range=[80, 100], title="% complet")
        st.plotly_chart(charts.apply_theme(fig, height=420), width='stretch')

    with right:
        st.subheader("Valeurs manquantes par champ")
        miss = pd.DataFrame({
            "Champ": list(qr.missing_by_column.keys()),
            "Manquants": list(qr.missing_by_column.values()),
        }).sort_values("Manquants", ascending=False)
        st.dataframe(miss, hide_index=True, width='stretch', height=300)

        st.subheader("Valeurs aberrantes détectées & bornées")
        if qr.outliers_by_column:
            ob = pd.DataFrame({"Champ": list(qr.outliers_by_column.keys()),
                               "Nombre": list(qr.outliers_by_column.values())})
            st.dataframe(ob, hide_index=True, width='stretch')
        else:
            st.success("Aucune valeur implausible détectée.")

    with st.expander("Ce que le pipeline vérifie"):
        st.markdown(
            "- **Schéma & libellés** — tables requises présentes, libellés de disponibilité valides\n"
            "- **Doublons** — les lignes joueur-jour identiques (double import) sont retirées\n"
            "- **Valeurs manquantes** — questionnaires de bien-être non remplis, GPS non porté\n"
            "- **Aberrations** — les valeurs hors bornes physiologiques plausibles sont "
            "signalées et bornées (sans destruction), jamais supprimées en silence\n"
            "- **Harmonisation des unités** — distances en mètres, temps en secondes\n"
            "- **Fraîcheur** — jours depuis la dernière observation")
    ui.disclaimer()
