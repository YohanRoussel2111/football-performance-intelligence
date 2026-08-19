"""Vue Direction — répond en moins de 30 secondes, pour un Head of Performance /
entraîneur : qui surveiller ? pourquoi ? qu'est-ce qui a changé ? quoi revoir ?
quelle confiance ?"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import app_core, ui
from src.ml import explain as ml_explain
from src.ml.inference import confidence_from_probability


def render():
    ui.header("Synthèse Direction", "qui surveiller — et pourquoi")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    snap = app_core.latest_snapshot(a["table"], scores, a["players"])
    expl = app_core.get_explainer(bundle["metadata"]["trained_at"])

    as_of = pd.to_datetime(snap["date"]).max().date()
    st.caption(f"État de l'effectif au **{as_of}**  ·  prochaine fenêtre de match à venir")

    # ---- indicateurs de tête ----
    avail = (snap["availability_status"] == "available").sum()
    modified = (snap["availability_status"] == "modified_training").sum()
    unavail = (snap["availability_status"] == "unavailable").sum()
    high = (snap["monitoring_level"] == "HIGH").sum()
    c = st.columns(4)
    c[0].metric("Disponibles", int(avail))
    c[1].metric("À surveiller (ÉLEVÉ)", int(high))
    c[2].metric("Aménagés", int(modified))
    c[3].metric("Indisponibles", int(unavail))

    st.divider()

    # ---- liste de revue prioritaire ----
    st.subheader("Liste de revue prioritaire")
    watch = snap[snap["availability_status"] == "available"].copy()
    watch = watch.sort_values("risk_probability", ascending=False).head(5)

    if watch.empty or watch["risk_probability"].max() < bundle["level_bands"]["moderate"]:
        st.success("Aucun joueur disponible ne dépasse actuellement le seuil de "
                   "surveillance. Profil de risque de l'effectif dans la norme.")
    for _, r in watch.iterrows():
        _player_card(r, a, expl, bundle)

    ui.disclaimer()


def _delta_text(a, pid, current_prob):
    """Ce qui a changé : risque vs il y a 7 jours."""
    scores = app_core.get_scores(_bundle_version=app_core.get_bundle()["metadata"]["trained_at"])
    s = scores[scores["player_id"] == pid].sort_values("date")
    if len(s) < 8:
        return "—"
    prev = s.iloc[-8]["risk_probability"]
    d = current_prob - prev
    arrow = "▲" if d > 0.02 else ("▼" if d < -0.02 else "▬")
    return f"{arrow} {d:+.0%} vs il y a 7 j"


def _player_card(r, a, expl, bundle):
    lvl = r["monitoring_level"]
    with st.container(border=True):
        cols = st.columns([2.4, 1.1, 1.2, 3.3])
        with cols[0]:
            st.markdown(f"**{r['player_name']}**  ·  {r['position']} · {int(r['age'])} ans")
            st.markdown(ui.chip(lvl, lvl), unsafe_allow_html=True)
        cols[1].metric("Risque", f"{r['risk_probability']:.0%}")
        conf = confidence_from_probability(r["risk_probability"])
        cols[2].metric("Confiance", f"{conf:.0f}%")
        with cols[3]:
            row = a["table"][(a["table"].player_id == r["player_id"]) &
                             (a["table"].date == r["date"])]
            st.caption(f"Ce qui a changé : {_delta_text(a, r['player_id'], r['risk_probability'])}")
            if not row.empty:
                e = ml_explain.explain_row(expl, row.iloc[0], top_n=3)
                factors = ml_explain.narrative_factors(e, max_factors=3)
                st.markdown("**Pourquoi :** " + "  ".join(
                    f"<span style='color:#e5484d'>{f}</span>" for f in factors),
                    unsafe_allow_html=True)
