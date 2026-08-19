"""Performance Actions — decision support. Generates cautious, non-medical
review prompts for flagged players. The tool recommends a staff review; it never
prescribes training or makes a medical/selection decision."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import app_core, ui
from src.ml import explain as ml_explain

# Factor -> concrete, non-prescriptive review prompt.
ACTION_LIBRARY = {
    "hsr_7d": "Review recent high-speed running exposure with the performance staff",
    "hsr_3d": "Review recent high-speed running exposure with the performance staff",
    "acwr": "Review acute:chronic load balance; consider load distribution this week",
    "load_week_change": "Review the week-on-week load increase",
    "load_7d": "Review accumulated 7-day load",
    "fatigue": "Discuss reported fatigue with the player",
    "fatigue_z": "Reported fatigue is elevated vs the player's own baseline — check in",
    "muscle_soreness": "Review muscle soreness with medical/physio staff",
    "muscle_soreness_z": "Soreness is elevated vs baseline — flag to physio",
    "cmj_pct_change": "Compare CMJ with the individual baseline; consider a re-test",
    "cmj_z": "Neuromuscular readiness (CMJ) is below baseline — consider a re-test",
    "cmj_last": "Consider a CMJ re-test to confirm neuromuscular status",
    "sleep_quality_z": "Review recent sleep quality",
    "sleep_duration": "Review recent sleep duration",
    "minutes_7d": "High recent match minutes — consider recovery emphasis",
    "minutes_14d": "High 14-day match minutes — consider fixture management",
    "matches_14d": "Fixture congestion is high — consider rotation options",
    "days_since_last_match": "Short recovery window since the last match",
    "monotony": "Training monotony is elevated — consider varying session load",
    "strain": "Training strain is elevated this week",
}
GENERIC = "Reassess before the next high-intensity session"


def render():
    ui.header("Performance Actions", "cautious, non-medical decision support")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    expl = app_core.get_explainer(bundle["metadata"]["trained_at"])
    snap = app_core.latest_snapshot(a["table"], scores, a["players"])

    st.info("These are **discussion prompts for the staff**, generated from the "
            "model's contributing factors. The system recommends a review — it "
            "does not decide whether a player trains, plays, or is injured.")

    flagged = snap[(snap["availability_status"] == "available") &
                   (snap["monitoring_level"].isin(["HIGH", "MODERATE"]))]
    flagged = flagged.sort_values("risk_probability", ascending=False)

    if flagged.empty:
        st.success("No available player currently requires escalated review.")
        ui.disclaimer()
        return

    for _, r in flagged.iterrows():
        row = a["table"][(a["table"].player_id == r["player_id"]) &
                         (a["table"].date == r["date"])].iloc[0]
        e = ml_explain.explain_row(expl, row, top_n=6)
        actions = _actions_from_factors(e)
        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(f"### {r['player_name']}  ·  {r['position']}")
                st.markdown(
                    f"Priority: {ui.chip(r['monitoring_level'], r['monitoring_level'])}"
                    f"  ·  risk {r['risk_probability']:.0%}",
                    unsafe_allow_html=True)
            top[1].metric("PMI", f"{r['pmi']:.0f}")
            st.markdown("**Recommended discussion:**")
            for act in actions:
                st.markdown(f"- {act}")
    ui.disclaimer()


def _actions_from_factors(explanation: dict) -> list[str]:
    seen, actions = set(), []
    for f in explanation["risk_increasing"]:
        text = ACTION_LIBRARY.get(f["feature"])
        if text and text not in seen:
            seen.add(text)
            actions.append(text)
        if len(actions) >= 4:
            break
    actions.append(GENERIC)
    return actions
