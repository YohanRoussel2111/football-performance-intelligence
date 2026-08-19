"""Performance du modèle — comparaison, validation temporelle, métriques, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import app_core, ui
from src.visualization import charts

_MODEL_FR = {"Logistic Regression": "Régression logistique",
             "Random Forest": "Forêt aléatoire", "XGBoost": "XGBoost",
             "Gradient Boosting": "Gradient Boosting"}


def render():
    ui.header("Performance du modèle", "validation temporelle, métriques & calibration")
    bundle = app_core.get_bundle()
    meta = bundle["metadata"]
    comp = bundle["comparison"]
    test = bundle["test_metrics"]

    c = st.columns(5)
    c[0].metric("Modèle retenu", _MODEL_FR.get(meta["selected_model"], meta["selected_model"]))
    c[1].metric("ROC-AUC (test)", f"{test['roc_auc']:.3f}")
    c[2].metric("PR-AUC (test)", f"{test['pr_auc']:.3f}")
    c[3].metric("Rappel", f"{test['recall']:.2f}")
    c[4].metric("Précision", f"{test['precision']:.2f}")

    st.caption(
        "Cible : joueur actuellement disponible ; épisode de disponibilité réduite "
        f"(aménagé/indisponible) dans les {meta['prediction_horizon_days']} prochains "
        f"jours.  ·  Taux de base (train) {meta['train_base_rate']:.1%}  ·  "
        f"n train/valid/test = {meta['n_train']}/{meta['n_valid']}/{meta['n_test']}")

    ui.science_box(
        "Comme les épisodes sont <b>rares</b>, on privilégie la <b>PR-AUC</b> (aire "
        "sous la courbe précision-rappel) plutôt que l'exactitude, trompeuse en cas de "
        "déséquilibre des classes (Saito &amp; Rehmsmeier, 2015). Le déséquilibre est "
        "géré au niveau de l'estimateur (class_weight / scale_pos_weight), sans "
        "rééchantillonnage, pour préserver la calibration.")

    st.divider()
    st.subheader("Comparaison des modèles (fenêtre de validation)")
    rows = []
    for name, m in comp.items():
        rows.append({
            "Modèle": _MODEL_FR.get(name, name) + (" ✓" if name == meta["selected_model"] else ""),
            "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
            "F1": m["f1"], "Rappel": m["recall"], "Précision": m["precision"],
            "PR-AUC (VC temporelle)": m["temporal_cv_pr_auc"], "Brier": m["brier"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch',
                 column_config={col: st.column_config.NumberColumn(format="%.3f")
                                for col in ["PR-AUC", "ROC-AUC", "F1", "Rappel",
                                            "Précision", "PR-AUC (VC temporelle)", "Brier"]})
    st.caption("La sélection se fait sur la PR-AUC en validation croisée temporelle — "
               "la métrique la plus pertinente pour les événements rares, estimée sur "
               "des plis à fenêtre glissante croissante.")

    left, right = st.columns(2)
    with left:
        st.subheader("Calibration")
        st.plotly_chart(
            charts.calibration_chart(test["calibration_curve"], test["brier"]),
            width='stretch')
        st.caption(f"La calibration a fait passer le score de Brier (test) de "
                   f"**{bundle['uncalibrated_test_brier']:.3f}** (brut) à "
                   f"**{test['brier']:.3f}** (calibré). Des probabilités calibrées "
                   "permettent au staff de lire un risque de 30 % comme *réellement* "
                   "~30 % du temps.")
    with right:
        st.subheader("Matrice de confusion (test)")
        st.plotly_chart(_confusion(test["confusion_matrix"]), width='stretch')
        st.caption(f"Au seuil opérationnel {test['threshold']:.2f}, réglé pour le "
                   "rappel : un outil de dépistage doit rarement manquer un cas réel, "
                   "quitte à accepter quelques fausses alertes vite écartées par le staff.")

    st.divider()
    with st.expander("Pourquoi une validation temporelle (et non un split aléatoire) ?"):
        st.markdown(
            "Dans un contexte longitudinal, chaque prédiction porte sur le *futur* à "
            "partir du *passé*. Un split aléatoire ferait fuiter de l'information "
            "future (et des mesures répétées d'un même joueur) dans l'entraînement, "
            "gonflant artificiellement le score. On découpe donc strictement par le "
            f"temps : les {app_core.config.TRAIN_FRACTION:.0%} premiers de la saison "
            f"entraînent, les {app_core.config.VALID_FRACTION:.0%} suivants valident/"
            "calibrent, et la dernière fenêtre teste. La baisse visible entre "
            "validation et test est le coût honnête de la non-stationnarité (la "
            "congestion des matchs déplace le taux de base au fil de la saison) — "
            "exactement ce qu'un déploiement rencontrerait.")


def _confusion(cm):
    cm = np.array(cm)
    labels = ["Aucun événement", "Événement"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Prédit : {l}" for l in labels], y=[f"Réel : {l}" for l in labels],
        colorscale="Teal", showscale=False,
        text=cm, texttemplate="%{text}", textfont=dict(size=20, color="#e7ecf4"),
        hovertemplate="%{y} / %{x} : %{z}<extra></extra>"))
    return charts.apply_theme(fig, height=360)
