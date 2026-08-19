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
    "load_3d": ("Recent 3-day training load", True),
    "load_7d": ("7-day accumulated load", True),
    "load_28d": ("28-day chronic load", False),
    "load_7d_mean": ("Average daily load (7d)", True),
    "load_7d_sd": ("Load variability (7d)", True),
    "hsr_3d": ("Recent high-speed running (3d)", True),
    "hsr_7d": ("High-speed running load (7d)", True),
    "monotony": ("Training monotony", True),
    "strain": ("Training strain", True),
    "acwr": ("Acute:chronic workload ratio", True),
    "acute_load": ("Acute load (7d)", True),
    "chronic_load": ("Chronic load (28d)", False),
    "load_week_change": ("Week-on-week load spike", True),
    "fatigue": ("Reported fatigue", True),
    "muscle_soreness": ("Muscle soreness", True),
    "stress": ("Reported stress", True),
    "sleep_duration": ("Sleep duration", False),
    "sleep_quality": ("Sleep quality", False),
    "fatigue_z": ("Fatigue vs individual baseline", True),
    "muscle_soreness_z": ("Soreness vs individual baseline", True),
    "stress_z": ("Stress vs individual baseline", True),
    "sleep_quality_z": ("Sleep quality vs baseline", False),
    "wellness_index_z": ("Overall wellness vs baseline", False),
    "cmj_last": ("Countermovement jump height", False),
    "cmj_pct_change": ("CMJ change vs individual baseline", False),
    "cmj_z": ("CMJ vs individual baseline (z)", False),
    "cmj_trend_7d": ("CMJ 7-day trend", False),
    "minutes_7d": ("Match minutes (last 7 days)", True),
    "minutes_14d": ("Match minutes (last 14 days)", True),
    "matches_14d": ("Matches in last 14 days", True),
    "matches_28d": ("Matches in last 28 days", True),
    "days_since_last_match": ("Days since last match", False),
    "age": ("Age", True),
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
