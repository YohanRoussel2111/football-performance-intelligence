"""AI Risk / Monitoring — squad risk ranking, per-player SHAP explanations,
global model drivers, and the DATA → SIGNAL → MODEL → EXPLANATION → ACTION story."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import app_core, ui
from src.visualization import charts
from src.ml import explain as ml_explain
from src.ml.inference import confidence_from_probability


def render():
    ui.header("AI Risk / Monitoring", "explainable early-warning across the squad")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    expl = app_core.get_explainer(bundle["metadata"]["trained_at"])
    snap = app_core.latest_snapshot(a["table"], scores, a["players"])

    tabs = st.tabs(["🚦 Squad risk", "🔬 Explain a player", "🌐 Model drivers"])

    with tabs[0]:
        st.caption("Available players ranked by predicted risk of a "
                   "reduced-availability event within the next 7 days.")
        rank = snap[snap["availability_status"] == "available"].copy()
        rank = rank.sort_values("risk_probability", ascending=False)
        show = pd.DataFrame({
            "Player": rank["player_name"], "Pos": rank["position"],
            "Level": rank["monitoring_level"].map(
                {"LOW": "🟢 LOW", "MODERATE": "🟡 MODERATE", "HIGH": "🔴 HIGH"}),
            "Risk": (rank["risk_probability"] * 100),
            "Confidence": rank["risk_probability"].map(
                lambda p: confidence_from_probability(p)),
            "ACWR": rank["acwr"], "Fatigue z": rank["fatigue_z"],
            "CMJ Δ%": rank["cmj_pct_change"] * 100,
        })
        st.dataframe(
            show, hide_index=True, width='stretch', height=560,
            column_config={
                "Risk": st.column_config.ProgressColumn(
                    "Risk", format="%.0f%%", min_value=0, max_value=100),
                "Confidence": st.column_config.NumberColumn(format="%.0f%%"),
                "ACWR": st.column_config.NumberColumn(format="%.2f"),
                "Fatigue z": st.column_config.NumberColumn(format="%+.2f"),
                "CMJ Δ%": st.column_config.NumberColumn(format="%+.1f%%"),
            })

    with tabs[1]:
        _explain_player(a, expl, bundle, scores, snap)

    with tabs[2]:
        st.caption("Mean absolute SHAP value over a sample of monitoring days — "
                   "the features the model relies on most across the squad.")
        model_rows = a["table"][a["table"].in_model == 1]
        sample = model_rows.sample(min(400, len(model_rows)), random_state=1)
        gi = ml_explain.global_importance(expl, sample)
        st.plotly_chart(charts.global_importance_chart(gi), width='stretch')

    ui.disclaimer()


def _explain_player(a, expl, bundle, scores, snap):
    opts = app_core.player_options(a["players"])
    default_pid = snap.sort_values("risk_probability", ascending=False).iloc[0]["player_id"]
    default_label = next((k for k, v in opts.items() if v == default_pid), list(opts)[0])
    label = st.selectbox("Player", list(opts), index=list(opts).index(default_label),
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
    c[0].metric("Risk probability", f"{ls['risk_probability']:.0%}")
    c[1].metric("Confidence", f"{confidence_from_probability(ls['risk_probability']):.0f}%")
    c[2].metric("PMI", f"{latest['pmi']:.0f}")

    left, right = st.columns([1.2, 1])
    with left:
        e = ml_explain.explain_row(expl, latest, top_n=8)
        st.plotly_chart(charts.shap_contribution_chart(e), width='stretch')
    with right:
        st.markdown("**DATA → SIGNAL → MODEL → EXPLANATION → ACTION**")
        st.markdown(f"- **Signal:** ACWR {latest['acwr']:.2f}, "
                    f"fatigue z {latest['fatigue_z']:+.2f}, "
                    f"CMJ {latest['cmj_pct_change']*100:+.1f}% vs baseline")
        st.markdown(f"- **Model:** {ls['monitoring_level']} "
                    f"(p={ls['risk_probability']:.0%})")
        st.markdown("- **Why:**")
        for f in e["risk_increasing"][:4]:
            st.markdown(f"  <span style='color:#e5484d'>+ {f['label']}</span>",
                        unsafe_allow_html=True)
        st.markdown("- **Action:** flag for staff review before next high-intensity "
                    "session (see *Performance Actions*).")
