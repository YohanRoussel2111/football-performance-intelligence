"""
Explainable AI layer.

Turns a model score into "here is *why* the model reached this conclusion":

    1. SHAP values give each feature's signed contribution for a specific
       player-day. We use the fast exact explainer for the model family
       (LinearExplainer for the logistic model, TreeExplainer for tree models).
    2. A translation layer maps raw feature contributions into practitioner
       language ("Reduced CMJ vs individual baseline", "Short recovery window"),
       grouping related features and keeping only the drivers that actually move
       this player's risk.

This is the difference between "the model says HIGH" and "the model says HIGH
because recent high-speed load is up, CMJ is 12% below this player's baseline,
and reported fatigue is elevated — please review."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import shap

# --------------------------------------------------------------------------- #
# Feature -> human-readable phrasing
# --------------------------------------------------------------------------- #
# Each entry: readable label + whether a HIGHER feature value means MORE risk.
FEATURE_META = {
    "load_3d": ("Charge d'entraînement récente (3 j)", True),
    "load_7d": ("Charge cumulée (7 j)", True),
    "load_28d": ("Charge chronique (28 j)", False),
    "load_7d_mean": ("Charge quotidienne moyenne (7 j)", True),
    "load_7d_sd": ("Variabilité de la charge (7 j)", True),
    "hsr_3d": ("Course haute vitesse récente (3 j)", True),
    "hsr_7d": ("Charge de course haute vitesse (7 j)", True),
    "monotony": ("Monotonie d'entraînement", True),
    "strain": ("Contrainte d'entraînement (strain)", True),
    "acwr": ("Ratio charge aiguë:chronique (ACWR)", True),
    "acute_load": ("Charge aiguë (7 j)", True),
    "chronic_load": ("Charge chronique (28 j)", False),
    "load_week_change": ("Pic de charge d'une semaine à l'autre", True),
    "fatigue": ("Fatigue déclarée", True),
    "muscle_soreness": ("Courbatures musculaires", True),
    "stress": ("Stress déclaré", True),
    "sleep_duration": ("Durée de sommeil", False),
    "sleep_quality": ("Qualité du sommeil", False),
    "fatigue_z": ("Fatigue vs référence individuelle", True),
    "muscle_soreness_z": ("Courbatures vs référence individuelle", True),
    "stress_z": ("Stress vs référence individuelle", True),
    "sleep_quality_z": ("Qualité du sommeil vs référence", False),
    "wellness_index_z": ("Bien-être global vs référence", False),
    "cmj_last": ("Hauteur de détente (CMJ)", False),
    "cmj_pct_change": ("Variation du CMJ vs référence individuelle", False),
    "cmj_z": ("CMJ vs référence individuelle (z)", False),
    "cmj_trend_7d": ("Tendance du CMJ (7 j)", False),
    "minutes_7d": ("Minutes de match (7 derniers jours)", True),
    "minutes_14d": ("Minutes de match (14 derniers jours)", True),
    "matches_14d": ("Matchs sur 14 jours", True),
    "matches_28d": ("Matchs sur 28 jours", True),
    "days_since_last_match": ("Jours depuis le dernier match", False),
    "age": ("Âge", True),
}


# --------------------------------------------------------------------------- #
# Explainer construction (fast exact per model family)
# --------------------------------------------------------------------------- #
def build_explainer(bundle: dict):
    model = bundle["base_model"]
    background = bundle["shap_background"]
    cols = bundle["feature_cols"]

    final = model[-1] if isinstance(model, Pipeline) else model
    if isinstance(final, LogisticRegression):
        pre = model[:-1] if isinstance(model, Pipeline) else None
        bg = pre.transform(background) if pre is not None else background.to_numpy()
        explainer = shap.LinearExplainer(final, bg)
        return {"kind": "linear", "explainer": explainer, "pre": pre, "cols": cols}

    # tree-based
    explainer = shap.TreeExplainer(final)
    return {"kind": "tree", "explainer": explainer, "pre": None, "cols": cols}


def shap_for_rows(expl: dict, X: pd.DataFrame) -> np.ndarray:
    """Return a (n_rows, n_features) array of signed SHAP contributions."""
    cols = expl["cols"]
    Xv = X[cols].astype(float).fillna(0.0)
    if expl["kind"] == "linear":
        Xt = expl["pre"].transform(Xv) if expl["pre"] is not None else Xv.to_numpy()
        sv = expl["explainer"].shap_values(Xt)
        return np.asarray(sv)
    # tree
    sv = expl["explainer"].shap_values(Xv)
    if isinstance(sv, list):          # older API returns list per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, classes)
        sv = sv[:, :, 1]
    return sv


# --------------------------------------------------------------------------- #
# Human-readable explanation
# --------------------------------------------------------------------------- #
def explain_row(expl: dict, feature_row: pd.Series, top_n: int = 5) -> dict:
    """Return top risk-increasing and risk-decreasing factors for one row."""
    cols = expl["cols"]
    X = feature_row[cols].to_frame().T
    sv = shap_for_rows(expl, X)[0]

    contribs = []
    for i, c in enumerate(cols):
        label, higher_is_worse = FEATURE_META.get(c, (c, True))
        contribs.append({
            "feature": c,
            "label": label,
            "shap": float(sv[i]),
            "value": float(feature_row[c]) if pd.notna(feature_row[c]) else np.nan,
            "direction": "increases" if sv[i] > 0 else "decreases",
        })
    contribs.sort(key=lambda d: abs(d["shap"]), reverse=True)

    increasing = [c for c in contribs if c["shap"] > 0][:top_n]
    decreasing = [c for c in contribs if c["shap"] < 0][:top_n]
    return {
        "top_factors": contribs[:top_n],
        "risk_increasing": increasing,
        "risk_decreasing": decreasing,
        "all": contribs,
    }


def narrative_factors(explanation: dict, max_factors: int = 5) -> list[str]:
    """Compact bullet phrases for the 'Why is this player flagged?' panel."""
    out = []
    for f in explanation["risk_increasing"][:max_factors]:
        out.append(f"+ {f['label']}")
    return out


def global_importance(expl: dict, X: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Mean |SHAP| across a sample of rows — the model's overall drivers."""
    cols = cols or expl["cols"]
    sv = shap_for_rows(expl, X)
    imp = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({
        "feature": cols,
        "label": [FEATURE_META.get(c, (c, True))[0] for c in cols],
        "mean_abs_shap": imp,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    from src.data_processing import pipeline
    from src.feature_engineering.features import build_features
    from src.ml import dataset as ds
    from src.ml.train import load_bundle
    from src.ml.inference import score_table, latest_scores

    out = pipeline.run_pipeline(persist=False)
    feats = build_features(out["daily"])
    lt = ds.build_learning_table(feats)
    bundle = load_bundle()

    expl = build_explainer(bundle)
    latest = latest_scores(bundle, lt)
    flagged = latest.sort_values("risk_probability", ascending=False).head(1)
    pid, pdate = flagged.iloc[0]["player_id"], flagged.iloc[0]["date"]
    row = lt[(lt.player_id == pid) & (lt.date == pdate)].iloc[0]
    e = explain_row(expl, row)
    print(f"Highest-risk player {pid} on {pdate.date()}: "
          f"p={flagged.iloc[0]['risk_probability']:.2f} "
          f"level={flagged.iloc[0]['monitoring_level']}")
    print("Why flagged:")
    for b in narrative_factors(e):
        print("  ", b)
    print("\nGlobal importance (top 8):")
    gi = global_importance(expl, lt[lt.in_model == 1].sample(300, random_state=1))
    print(gi.head(8).to_string(index=False))
