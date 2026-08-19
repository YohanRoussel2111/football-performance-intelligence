"""Actions de performance — aide à la décision. Génère des points de revue
prudents et non médicaux pour les joueurs signalés. L'outil recommande une revue
par le staff ; il ne prescrit jamais l'entraînement et ne pose aucun diagnostic."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import app_core, ui
from src.ml import explain as ml_explain

# Variable -> point de revue concret et non prescriptif.
ACTION_LIBRARY = {
    "hsr_7d": "Revoir l'exposition récente à la course haute vitesse avec le staff performance",
    "hsr_3d": "Revoir l'exposition récente à la course haute vitesse avec le staff performance",
    "acwr": "Revoir l'équilibre charge aiguë:chronique ; envisager une répartition de la charge cette semaine",
    "load_week_change": "Revoir l'augmentation de charge d'une semaine à l'autre",
    "load_7d": "Revoir la charge cumulée sur 7 jours",
    "fatigue": "Échanger avec le joueur sur la fatigue déclarée",
    "fatigue_z": "Fatigue élevée vs sa propre référence — prendre des nouvelles",
    "muscle_soreness": "Revoir les courbatures avec le staff médical/kiné",
    "muscle_soreness_z": "Courbatures élevées vs référence — signaler au kiné",
    "cmj_pct_change": "Comparer le CMJ à la référence individuelle ; envisager un re-test",
    "cmj_z": "Disponibilité neuromusculaire (CMJ) sous la référence — envisager un re-test",
    "cmj_last": "Envisager un re-test du CMJ pour confirmer le statut neuromusculaire",
    "sleep_quality_z": "Revoir la qualité de sommeil récente",
    "sleep_duration": "Revoir la durée de sommeil récente",
    "minutes_7d": "Minutes de match récentes élevées — envisager de mettre l'accent sur la récupération",
    "minutes_14d": "Minutes de match sur 14 jours élevées — envisager la gestion des temps de jeu",
    "matches_14d": "Congestion des matchs élevée — envisager des options de rotation",
    "days_since_last_match": "Fenêtre de récupération courte depuis le dernier match",
    "monotony": "Monotonie d'entraînement élevée — envisager de varier la charge des séances",
    "strain": "Contrainte d'entraînement élevée cette semaine",
}
GENERIC = "Réévaluer avant la prochaine séance à haute intensité"


def render():
    ui.header("Actions de performance", "aide à la décision prudente, non médicale")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    expl = app_core.get_explainer(bundle["metadata"]["trained_at"])
    snap = app_core.latest_snapshot(a["table"], scores, a["players"])

    st.info("Ce sont des **points de discussion pour le staff**, générés à partir "
            "des facteurs contributifs du modèle. Le système recommande une revue — "
            "il ne décide pas si un joueur s'entraîne, joue ou est blessé.")

    flagged = snap[(snap["availability_status"] == "available") &
                   (snap["monitoring_level"].isin(["HIGH", "MODERATE"]))]
    flagged = flagged.sort_values("risk_probability", ascending=False)

    if flagged.empty:
        st.success("Aucun joueur disponible ne nécessite actuellement une revue renforcée.")
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
                    f"Priorité : {ui.chip(r['monitoring_level'], r['monitoring_level'])}"
                    f"  ·  risque {r['risk_probability']:.0%}",
                    unsafe_allow_html=True)
            top[1].metric("IPM", f"{r['pmi']:.0f}")
            st.markdown("**Discussion recommandée :**")
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
