"""Player Profile — detailed longitudinal view for one player, plus the AI
monitoring panel with SHAP explanation."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import app_core, ui
from src.visualization import charts
from src.ml import explain as ml_explain
from src.ml.inference import confidence_from_probability


def render():
    ui.header("Player Profile", "individual longitudinal monitoring")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    expl = app_core.get_explainer(bundle["metadata"]["trained_at"])

    opts = app_core.player_options(a["players"])
    # default to the highest-risk player for a compelling demo
    snap = app_core.latest_snapshot(a["table"], scores, a["players"])
    default_pid = snap.sort_values("risk_probability", ascending=False).iloc[0]["player_id"]
    default_label = next((k for k, v in opts.items() if v == default_pid), list(opts)[0])
    label = st.selectbox("Select player", list(opts),
                         index=list(opts).index(default_label))
    pid = opts[label]

    pdata = a["players"][a["players"].player_id == pid].iloc[0]
    ptable = a["table"][a["table"].player_id == pid].sort_values("date")
    pscore = scores[scores.player_id == pid].sort_values("date")
    latest = ptable.iloc[-1]
    latest_score = pscore.iloc[-1]

    # ---- header info ----
    c = st.columns(5)
    c[0].metric("Position", pdata["position"])
    c[1].metric("Age", f"{int(pdata['age'])}")
    c[2].metric("Dominant leg", pdata["dominant_leg"])
    c[3].metric("Availability", latest["availability_status"].replace("_", " ").title())
    c[4].metric("PMI", f"{latest['pmi']:.0f}", help="Performance Monitoring Index (0–100).")

    tabs = st.tabs(["📈 Load", "😴 Recovery", "🦵 Neuromuscular",
                    "🩹 Availability", "🤖 AI Monitoring"])

    with tabs[0]:
        st.plotly_chart(charts.load_trend_chart(ptable), width='stretch')
        st.plotly_chart(charts.acwr_chart(ptable), width='stretch')

    with tabs[1]:
        st.plotly_chart(charts.wellness_chart(ptable), width='stretch')
        cc = st.columns(4)
        cc[0].metric("Sleep (7d)", f"{ptable['sleep_duration'].tail(7).mean():.1f} h")
        cc[1].metric("Fatigue (z)", f"{latest['fatigue_z']:+.2f}")
        cc[2].metric("Soreness (z)", f"{latest['muscle_soreness_z']:+.2f}")
        cc[3].metric("Stress (z)", f"{latest['stress_z']:+.2f}")

    with tabs[2]:
        st.plotly_chart(charts.cmj_chart(ptable), width='stretch')
        cc = st.columns(3)
        cc[0].metric("CMJ vs baseline", f"{latest['cmj_pct_change']*100:+.1f}%")
        cc[1].metric("CMJ (z)", f"{latest['cmj_z']:+.2f}")
        cc[2].metric("Neuromuscular", latest["neuromuscular_status"])

    with tabs[3]:
        _availability_tab(a, pid)

    with tabs[4]:
        _ai_tab(a, expl, bundle, ptable, pscore, latest, latest_score)

    ui.disclaimer()


def _availability_tab(a, pid):
    eps = a["episodes"]
    eps = eps[eps.player_id == pid].copy() if len(eps) else eps
    st.subheader("Availability history")
    if eps is None or len(eps) == 0:
        st.success("No reduced-availability episodes recorded this season.")
        return
    eps = eps.sort_values("start_date")
    show = pd.DataFrame({
        "Type": eps["type"].map({"modified": "🟡 Modified training",
                                 "unavailable": "🔴 Unavailable"}),
        "Start": pd.to_datetime(eps["start_date"]).dt.date,
        "End": pd.to_datetime(eps["end_date"]).dt.date,
        "Days": eps["duration_days"],
    })
    st.dataframe(show, hide_index=True, width='stretch')
    total = int(eps.loc[eps.type == "unavailable", "duration_days"].sum())
    st.metric("Days unavailable (season)", total)


def _ai_tab(a, expl, bundle, ptable, pscore, latest, latest_score):
    left, right = st.columns([1, 1.25])
    prob = latest_score["risk_probability"]
    lvl = latest_score["monitoring_level"]
    with left:
        st.markdown("#### Current monitoring level")
        st.markdown(ui.chip(lvl, lvl), unsafe_allow_html=True)
        m = st.columns(2)
        m[0].metric("Probability", f"{prob:.0%}")
        m[1].metric("Confidence", f"{confidence_from_probability(prob):.0f}%")
        st.caption(f"Horizon: reduced-availability event within "
                   f"{bundle['metadata']['prediction_horizon_days']} days. "
                   f"Model: {bundle['metadata']['selected_model']} "
                   f"v{bundle['metadata']['model_version']}.")
        st.plotly_chart(charts.risk_trajectory_chart(pscore, bundle["level_bands"]),
                        width='stretch')
    with right:
        st.markdown("#### Why is this player flagged?")
        e = ml_explain.explain_row(expl, latest, top_n=8)
        st.plotly_chart(charts.shap_contribution_chart(e), width='stretch')
        st.markdown("**Main contributing factors**")
        for f in e["risk_increasing"][:5]:
            st.markdown(
                f"<span style='color:#e5484d'>+ {f['label']}</span> "
                f"<span style='color:#8ea0bd'>(value {f['value']:.2f})</span>",
                unsafe_allow_html=True)
