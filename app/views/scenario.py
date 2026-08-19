"""Simulateur de scénarios — « what-if » exploratoire. Le staff ajuste charge,
minutes, sommeil, course haute vitesse et récupération, et le modèle re-score.
Il s'agit d'une exploration de sensibilité du modèle, PAS d'inférence causale :
on montre comment le modèle réagirait à un autre vecteur d'entrée, pas ce qui
arriverait physiquement au joueur."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import app_core, ui
from src.visualization import charts
from src.ml.inference import monitoring_level


def render():
    ui.header("Simulateur de scénarios", "« what-if » exploratoire sur le modèle")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    snap = app_core.latest_snapshot(a["table"], scores, a["players"])

    st.warning("Exploratoire uniquement. Ajuster une entrée recalcule le score du "
               "modèle ; cela ne **prédit pas** l'effet physiologique d'un changement "
               "d'entraînement. Corrélation ≠ causalité.")

    opts = app_core.player_options(a["players"])
    default_pid = snap.sort_values("risk_probability", ascending=False).iloc[0]["player_id"]
    default_label = next((k for k, v in opts.items() if v == default_pid), list(opts)[0])
    label = st.selectbox("Joueur", list(opts), index=list(opts).index(default_label))
    pid = opts[label]
    base = a["table"][a["table"].player_id == pid].sort_values("date").iloc[-1]

    st.markdown("#### Ajuster le scénario de la semaine à venir")
    c = st.columns(5)
    load_mult = c[0].slider("Charge d'entraînement", 0.5, 1.5, 1.0, 0.05,
                            help="Met à l'échelle les variables de charge récente.")
    minutes = c[1].slider("Minutes prochain match", 0, 95, int(min(base["minutes_played"] or 90, 95)), 5)
    hsr_mult = c[2].slider("Course haute vitesse", 0.5, 1.5, 1.0, 0.05)
    sleep = c[3].slider("Sommeil (h/nuit)", 5.0, 9.5, float(round(base["sleep_duration"], 1)), 0.25)
    fatigue = c[4].slider("Fatigue déclarée (1–7)", 1, 7, int(base["fatigue"]))

    cols = bundle["feature_cols"]
    scenario = _apply_scenario(base, cols, load_mult, minutes, hsr_mult, sleep, fatigue,
                               a["table"], pid)

    model = bundle["calibrated_model"]
    p_base = model.predict_proba(base[cols].astype(float).fillna(0).to_frame().T)[0, 1]
    p_scn = model.predict_proba(scenario[cols].astype(float).fillna(0).to_frame().T)[0, 1]
    lvl_base = monitoring_level(p_base, bundle["level_bands"])
    lvl_scn = monitoring_level(p_scn, bundle["level_bands"])

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Scénario A — actuel")
        st.markdown(ui.chip(lvl_base, lvl_base), unsafe_allow_html=True)
        st.metric("Risque modèle", f"{p_base:.0%}")
    with right:
        st.markdown("#### Scénario B — ajusté")
        st.markdown(ui.chip(lvl_scn, lvl_scn), unsafe_allow_html=True)
        st.metric("Risque modèle", f"{p_scn:.0%}", delta=f"{(p_scn - p_base):+.0%}",
                  delta_color="inverse")

    st.plotly_chart(_compare_chart(base, scenario), width='stretch')
    ui.disclaimer()


def _apply_scenario(base, cols, load_mult, minutes, hsr_mult, sleep, fatigue, table, pid):
    s = base.copy()
    for c in ["load_3d", "load_7d", "load_7d_mean", "acute_load", "strain"]:
        if c in cols:
            s[c] = base[c] * load_mult
    for c in ["hsr_3d", "hsr_7d"]:
        if c in cols:
            s[c] = base[c] * hsr_mult
    if base.get("chronic_load", 0) and "acwr" in cols:
        s["acwr"] = np.clip((base["acute_load"] * load_mult) / max(base["chronic_load"], 1), 0, 3)
    added = minutes - (base["minutes_played"] or 0)
    if "minutes_7d" in cols:
        s["minutes_7d"] = max(0, base["minutes_7d"] + added)
    if "minutes_14d" in cols:
        s["minutes_14d"] = max(0, base["minutes_14d"] + added)
    if "sleep_duration" in cols:
        s["sleep_duration"] = sleep
    if "sleep_quality_z" in cols and "sleep_duration_baseline" in base:
        base_sleep = base.get("sleep_duration_baseline", sleep)
        s["sleep_quality_z"] = np.clip((sleep - base_sleep) / 0.6, -4, 4)
    if "fatigue" in cols:
        s["fatigue"] = fatigue
    if "fatigue_z" in cols and "fatigue_baseline" in base:
        s["fatigue_z"] = np.clip((fatigue - base["fatigue_baseline"]) /
                                 max(base.get("fatigue", 1) * 0 + 1.0, 0.5), -4, 4)
    return s


def _compare_chart(base, scenario):
    labels = ["Charge 7 j", "HSR 7 j", "ACWR", "Minutes 7 j", "Sommeil", "Fatigue"]
    keys = ["load_7d", "hsr_7d", "acwr", "minutes_7d", "sleep_duration", "fatigue"]
    a_vals, b_vals = [], []
    for k in keys:
        bv = float(base.get(k, 0)) or 1e-9
        a_vals.append(1.0)
        b_vals.append(float(scenario.get(k, 0)) / bv)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=a_vals, name="A (actuel)", marker_color=charts.NEUTRAL))
    fig.add_trace(go.Bar(x=labels, y=b_vals, name="B (ajusté)", marker_color=charts.PRIMARY))
    fig.update_yaxes(title="relatif à l'actuel")
    fig.update_layout(barmode="group")
    return charts.apply_theme(fig, title="Comparaison des indicateurs (A vs B)", height=340)
