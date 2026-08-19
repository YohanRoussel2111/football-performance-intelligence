"""Profil Joueur — vue longitudinale détaillée pour un joueur, plus le panneau
de suivi IA avec explication SHAP."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import app_core, ui
from src.visualization import charts
from src.ml import explain as ml_explain
from src.ml.inference import confidence_from_probability


def render():
    ui.header("Profil Joueur", "suivi longitudinal individuel")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    expl = app_core.get_explainer(bundle["metadata"]["trained_at"])

    opts = app_core.player_options(a["players"])
    snap = app_core.latest_snapshot(a["table"], scores, a["players"])
    default_pid = snap.sort_values("risk_probability", ascending=False).iloc[0]["player_id"]
    default_label = next((k for k, v in opts.items() if v == default_pid), list(opts)[0])
    label = st.selectbox("Sélectionner un joueur", list(opts),
                         index=list(opts).index(default_label))
    pid = opts[label]

    pdata = a["players"][a["players"].player_id == pid].iloc[0]
    ptable = a["table"][a["table"].player_id == pid].sort_values("date")
    pscore = scores[scores.player_id == pid].sort_values("date")
    latest = ptable.iloc[-1]
    latest_score = pscore.iloc[-1]

    # ---- fiche d'identité ----
    c = st.columns(5)
    c[0].metric("Poste", pdata["position"])
    c[1].metric("Âge", f"{int(pdata['age'])}")
    c[2].metric("Pied fort", "Droit" if pdata["dominant_leg"] == "Right" else "Gauche")
    c[3].metric("Disponibilité", ui.FR_AVAIL.get(latest["availability_status"],
                                                 latest["availability_status"]))
    c[4].metric("IPM", f"{latest['pmi']:.0f}",
                help="Indice de Performance & Monitoring (0–100).")

    tabs = st.tabs(["📈 Charge", "😴 Récupération", "🦵 Neuromusculaire",
                    "🩹 Disponibilité", "🤖 Suivi IA"])

    with tabs[0]:
        ui.science_box(
            "La <b>charge aiguë (7 j)</b> comparée à la <b>charge chronique (28 j)</b> "
            "traduit l'équilibre entre fatigue et forme (fitness-fatigue). Un ACWR "
            "durablement au-dessus de ~1,3 est associé, dans la littérature, à un "
            "risque accru — à interpréter avec prudence (Gabbett, 2016 ; Impellizzeri, 2020).")
        st.plotly_chart(charts.load_trend_chart(ptable), width='stretch')
        st.plotly_chart(charts.acwr_chart(ptable), width='stretch')

    with tabs[1]:
        ui.science_box(
            "Le <b>bien-être auto-déclaré</b> (fatigue, courbatures, sommeil, stress) "
            "est un marqueur sensible et peu coûteux de la réponse à la charge "
            "(Hooper &amp; Mackinnon, 1995 ; Saw et al., 2016). On le lit ici en "
            "<b>écart z</b> par rapport à la référence propre au joueur.")
        st.plotly_chart(charts.wellness_chart(ptable), width='stretch')
        cc = st.columns(4)
        cc[0].metric("Sommeil (7 j)", f"{ptable['sleep_duration'].tail(7).mean():.1f} h")
        cc[1].metric("Fatigue (z)", f"{latest['fatigue_z']:+.2f}")
        cc[2].metric("Courbatures (z)", f"{latest['muscle_soreness_z']:+.2f}")
        cc[3].metric("Stress (z)", f"{latest['stress_z']:+.2f}")

    with tabs[2]:
        ui.science_box(
            "La <b>détente verticale (CMJ)</b> est un test de terrain validé de la "
            "fatigue neuromusculaire : une baisse marquée vs la référence individuelle "
            "signale une récupération incomplète (Claudino et al., 2017 ; Gathercole "
            "et al., 2015).")
        st.plotly_chart(charts.cmj_chart(ptable), width='stretch')
        cc = st.columns(3)
        cc[0].metric("CMJ vs référence", f"{latest['cmj_pct_change']*100:+.1f}%")
        cc[1].metric("CMJ (z)", f"{latest['cmj_z']:+.2f}")
        cc[2].metric("Statut neuromusc.", ui.FR_NM.get(latest["neuromuscular_status"],
                                                       latest["neuromuscular_status"]))

    with tabs[3]:
        _availability_tab(a, pid)

    with tabs[4]:
        _ai_tab(a, expl, bundle, ptable, pscore, latest, latest_score)

    ui.disclaimer()


def _availability_tab(a, pid):
    eps = a["episodes"]
    eps = eps[eps.player_id == pid].copy() if len(eps) else eps
    st.subheader("Historique de disponibilité")
    if eps is None or len(eps) == 0:
        st.success("Aucun épisode de disponibilité réduite enregistré cette saison.")
        return
    eps = eps.sort_values("start_date")
    show = pd.DataFrame({
        "Type": eps["type"].map({"modified": "🟡 Entraînement aménagé",
                                 "unavailable": "🔴 Indisponible"}),
        "Début": pd.to_datetime(eps["start_date"]).dt.date,
        "Fin": pd.to_datetime(eps["end_date"]).dt.date,
        "Jours": eps["duration_days"],
    })
    st.dataframe(show, hide_index=True, width='stretch')
    total = int(eps.loc[eps.type == "unavailable", "duration_days"].sum())
    st.metric("Jours d'indisponibilité (saison)", total)


def _ai_tab(a, expl, bundle, ptable, pscore, latest, latest_score):
    left, right = st.columns([1, 1.25])
    prob = latest_score["risk_probability"]
    lvl = latest_score["monitoring_level"]
    with left:
        st.markdown("#### Niveau de surveillance actuel")
        st.markdown(ui.chip(lvl, lvl), unsafe_allow_html=True)
        m = st.columns(2)
        m[0].metric("Probabilité", f"{prob:.0%}")
        m[1].metric("Confiance", f"{confidence_from_probability(prob):.0f}%")
        st.caption(f"Horizon : épisode de disponibilité réduite dans les "
                   f"{bundle['metadata']['prediction_horizon_days']} jours. "
                   f"Modèle : {bundle['metadata']['selected_model']} "
                   f"v{bundle['metadata']['model_version']}.")
        st.plotly_chart(charts.risk_trajectory_chart(pscore, bundle["level_bands"]),
                        width='stretch')
    with right:
        st.markdown("#### Pourquoi ce joueur est-il signalé ?")
        e = ml_explain.explain_row(expl, latest, top_n=8)
        st.plotly_chart(charts.shap_contribution_chart(e), width='stretch')
        st.markdown("**Principaux facteurs contributifs**")
        for f in e["risk_increasing"][:5]:
            st.markdown(
                f"<span style='color:#e5484d'>+ {f['label']}</span> "
                f"<span style='color:#8ea0bd'>(valeur {f['value']:.2f})</span>",
                unsafe_allow_html=True)
