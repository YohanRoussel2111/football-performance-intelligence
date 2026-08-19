"""
Football Performance Intelligence — point d'entrée Streamlit.

Lancer avec :  streamlit run app/main.py    (ou :  python run_app.py)
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
    actions, scenario, model_perf, model_monitor, data_quality, science,
)


def _demo_controls():
    st.sidebar.markdown("### ⚡ Contrôles démo")
    if st.sidebar.button("1 · Charger le jeu de démo", width='stretch'):
        app_core.get_analysis.clear()
        with st.spinner("Génération de la saison synthétique & pipeline…"):
            a = app_core.get_analysis()
        st.sidebar.success(
            f"{a['quality_report'].n_players} joueurs · "
            f"{a['quality_report'].n_observations:,} obs chargées")
    if st.sidebar.button("2 · Lancer le modèle", width='stretch'):
        with st.spinner("Entraînement & validation temporelle…"):
            b = app_core.retrain_model()
        st.sidebar.success(
            f"{b['metadata']['selected_model']} · "
            f"ROC-AUC (test) {b['test_metrics']['roc_auc']:.2f}")
    if st.sidebar.button("3 · Générer les prédictions", width='stretch'):
        with st.spinner("Scoring de l'effectif & écriture des prédictions…"):
            n = _write_predictions()
        st.sidebar.success(f"{n} prédictions écrites dans SQLite")
    st.sidebar.caption("Puis ouvrir **Risque IA / Suivi → Expliquer un joueur**.")


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
        "Centre de commande": [
            st.Page(executive.render, title="Synthèse Direction", icon="🎯", default=True,
                    url_path="executive"),
            st.Page(squad.render, title="Vue d'ensemble", icon="👥", url_path="squad"),
        ],
        "Joueur & charge": [
            st.Page(player.render, title="Profil Joueur", icon="👤", url_path="player"),
            st.Page(load_monitoring.render, title="Suivi de la charge", icon="📈",
                    url_path="load"),
            st.Page(availability.render, title="Disponibilité", icon="🩹",
                    url_path="availability"),
        ],
        "Intelligence": [
            st.Page(ai_risk.render, title="Risque IA / Suivi", icon="🤖",
                    url_path="ai-risk"),
            st.Page(actions.render, title="Actions de performance", icon="✅",
                    url_path="actions"),
            st.Page(scenario.render, title="Simulateur de scénarios", icon="🧪",
                    url_path="scenario"),
        ],
        "Modèle & science": [
            st.Page(model_perf.render, title="Performance du modèle", icon="📊",
                    url_path="model-performance"),
            st.Page(model_monitor.render, title="Suivi du modèle", icon="🛰️",
                    url_path="model-monitoring"),
            st.Page(data_quality.render, title="Qualité des données", icon="🧹",
                    url_path="data-quality"),
            st.Page(science.render, title="Science & Méthodologie", icon="🔬",
                    url_path="science"),
        ],
    }
    nav = st.navigation(pages)
    _demo_controls()
    nav.run()


if __name__ == "__main__":
    main()
