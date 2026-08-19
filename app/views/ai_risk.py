"""Risque IA / Suivi — classement du risque de l'effectif, explications SHAP par
joueur, moteurs globaux du modèle, et le fil DONNÉES → SIGNAL → MODÈLE →
EXPLICATION → ACTION."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import app_core, ui
from src.visualization import charts
from src.ml import explain as ml_explain
from src.ml.inference import confidence_from_probability


def render():
    ui.header("Risque IA / Suivi", "alerte précoce explicable sur tout l'effectif")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    expl = app_core.get_explainer(bundle["metadata"]["trained_at"])
    snap = app_core.latest_snapshot(a["table"], scores, a["players"])

    tabs = st.tabs(["🚦 Risque de l'effectif", "🔬 Expliquer un joueur",
                    "🌐 Moteurs du modèle"])

    with tabs[0]:
        st.caption("Joueurs disponibles classés par risque prédit d'un épisode de "
                   "disponibilité réduite dans les 7 prochains jours.")
        rank = snap[snap["availability_status"] == "available"].copy()
        rank = rank.sort_values("risk_probability", ascending=False)
        show = pd.DataFrame({
            "Joueur": rank["player_name"], "Poste": rank["position"],
            "Niveau": rank["monitoring_level"].map(ui.fr_level_tag),
            "Risque": (rank["risk_probability"] * 100),
            "Confiance": rank["risk_probability"].map(
                lambda p: confidence_from_probability(p)),
            "ACWR": rank["acwr"], "Fatigue z": rank["fatigue_z"],
            "CMJ Δ%": rank["cmj_pct_change"] * 100,
        })
        st.dataframe(
            show, hide_index=True, width='stretch', height=560,
            column_config={
                "Risque": st.column_config.ProgressColumn(
                    "Risque", format="%.0f%%", min_value=0, max_value=100),
                "Confiance": st.column_config.NumberColumn(format="%.0f%%"),
                "ACWR": st.column_config.NumberColumn(format="%.2f"),
                "Fatigue z": st.column_config.NumberColumn(format="%+.2f"),
                "CMJ Δ%": st.column_config.NumberColumn(format="%+.1f%%"),
            })

    with tabs[1]:
        _explain_player(a, expl, bundle, scores, snap)

    with tabs[2]:
        st.caption("Valeur SHAP absolue moyenne sur un échantillon de jours de suivi "
                   "— les variables sur lesquelles le modèle s'appuie le plus.")
        model_rows = a["table"][a["table"].in_model == 1]
        sample = model_rows.sample(min(400, len(model_rows)), random_state=1)
        gi = ml_explain.global_importance(expl, sample)
        st.plotly_chart(charts.global_importance_chart(gi), width='stretch')

    ui.disclaimer()


def _explain_player(a, expl, bundle, scores, snap):
    opts = app_core.player_options(a["players"])
    default_pid = snap.sort_values("risk_probability", ascending=False).iloc[0]["player_id"]
    default_label = next((k for k, v in opts.items() if v == default_pid), list(opts)[0])
    label = st.selectbox("Joueur", list(opts), index=list(opts).index(default_label),
                         key="ai_explain_player")
    pid = opts[label]
    ptable = a["table"][a["table"].player_id == pid].sort_values("date")
    pscore = scores[scores.player_id == pid].sort_values("date")
    latest = ptable.iloc[-1]
    ls = pscore.iloc[-1]

    st.markdown(
        f"### {label.split('·')[1].strip()} — "
        f"{ui.chip(ls['monitoring_level'], ls['monitoring_level'])}",
        unsafe_allow_html=True)
    c = st.columns(3)
    c[0].metric("Probabilité de risque", f"{ls['risk_probability']:.0%}")
    c[1].metric("Confiance", f"{confidence_from_probability(ls['risk_probability']):.0f}%")
    c[2].metric("IPM", f"{latest['pmi']:.0f}")

    left, right = st.columns([1.2, 1])
    with left:
        e = ml_explain.explain_row(expl, latest, top_n=8)
        st.plotly_chart(charts.shap_contribution_chart(e), width='stretch')
    with right:
        st.markdown("**DONNÉES → SIGNAL → MODÈLE → EXPLICATION → ACTION**")
        st.markdown(f"- **Signal :** ACWR {latest['acwr']:.2f}, "
                    f"fatigue z {latest['fatigue_z']:+.2f}, "
                    f"CMJ {latest['cmj_pct_change']*100:+.1f}% vs référence")
        st.markdown(f"- **Modèle :** {ui.FR_LEVEL.get(ls['monitoring_level'], ls['monitoring_level'])} "
                    f"(p={ls['risk_probability']:.0%})")
        st.markdown("- **Pourquoi :**")
        for f in e["risk_increasing"][:4]:
            st.markdown(f"  <span style='color:#e5484d'>+ {f['label']}</span>",
                        unsafe_allow_html=True)
        st.markdown("- **Action :** à signaler pour revue par le staff avant la "
                    "prochaine séance à haute intensité (voir *Actions de performance*).")
