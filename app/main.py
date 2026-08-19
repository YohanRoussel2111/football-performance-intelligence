"""
Football Performance Intelligence — Streamlit entry point.

Run with:  streamlit run app/main.py    (or:  python run_app.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Football Performance Intelligence",
    page_icon="⚽", layout="wide", initial_sidebar_state="expanded",
)

from app import app_core, ui                      # noqa: E402
from app.views import (                           # noqa: E402
    executive, squad, player, load_monitoring, availability, ai_risk,
    actions, scenario, model_perf, model_monitor, data_quality,
)


def _demo_controls():
    st.sidebar.markdown("### ⚡ Demo controls")
    if st.sidebar.button("1 · Load Demo Dataset", width='stretch'):
        app_core.get_analysis.clear()
        with st.spinner("Generating synthetic season & running pipeline…"):
            a = app_core.get_analysis()
        st.sidebar.success(
            f"{a['quality_report'].n_players} players · "
            f"{a['quality_report'].n_observations:,} obs loaded")
    if st.sidebar.button("2 · Run Model", width='stretch'):
        with st.spinner("Training & temporally validating models…"):
            b = app_core.retrain_model()
        st.sidebar.success(
            f"{b['metadata']['selected_model']} · "
            f"test ROC-AUC {b['test_metrics']['roc_auc']:.2f}")
    if st.sidebar.button("3 · Generate Predictions", width='stretch'):
        with st.spinner("Scoring squad & writing predictions…"):
            n = _write_predictions()
        st.sidebar.success(f"{n} predictions written to SQLite")
    st.sidebar.caption("Then open **AI Risk / Monitoring → Explain a player**.")


def _write_predictions() -> int:
    from src.data_processing import database as db
    from datetime import datetime, timezone
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    latest = scores.sort_values("date").groupby("player_id").tail(1).copy()
    latest["model_version"] = bundle["metadata"]["model_version"]
    latest["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.write_table("predictions", latest[["player_id", "date", "model_version",
                   "risk_probability", "monitoring_level", "generated_at"]])
    return len(latest)


def main():
    ui.inject_css()
    ui.brand_sidebar()

    pages = {
        "Command centre": [
            st.Page(executive.render, title="Executive Brief", icon="🎯", default=True,
                    url_path="executive"),
            st.Page(squad.render, title="Squad Overview", icon="👥", url_path="squad"),
        ],
        "Player & load": [
            st.Page(player.render, title="Player Profile", icon="👤", url_path="player"),
            st.Page(load_monitoring.render, title="Load Monitoring", icon="📈",
                    url_path="load"),
            st.Page(availability.render, title="Availability", icon="🩹",
                    url_path="availability"),
        ],
        "Intelligence": [
            st.Page(ai_risk.render, title="AI Risk / Monitoring", icon="🤖",
                    url_path="ai-risk"),
            st.Page(actions.render, title="Performance Actions", icon="✅",
                    url_path="actions"),
            st.Page(scenario.render, title="Scenario Simulator", icon="🧪",
                    url_path="scenario"),
        ],
        "Model & data ops": [
            st.Page(model_perf.render, title="Model Performance", icon="📊",
                    url_path="model-performance"),
            st.Page(model_monitor.render, title="Model Monitoring", icon="🛰️",
                    url_path="model-monitoring"),
            st.Page(data_quality.render, title="Data Quality", icon="🧹",
                    url_path="data-quality"),
        ],
    }
    nav = st.navigation(pages)
    _demo_controls()
    nav.run()


if __name__ == "__main__":
    main()
