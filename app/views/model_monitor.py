"""Model Monitoring — production-style health checks: metadata, feature drift,
prediction distribution, data freshness, and a simple degradation flag."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import app_core, ui
from src.visualization import charts
from src.ml.explain import FEATURE_META


def render():
    ui.header("Model Monitoring", "is the model still healthy in production?")
    a = app_core.get_analysis()
    bundle = app_core.get_bundle()
    scores = app_core.get_scores(_bundle_version=bundle["metadata"]["trained_at"])
    meta = bundle["metadata"]
    qr = a["quality_report"]

    c = st.columns(5)
    c[0].metric("Model version", meta["model_version"])
    c[1].metric("Trained", meta["trained_at"][:10])
    c[2].metric("Train obs", f"{meta['n_train']:,}")
    c[3].metric("Features", meta["n_features"])
    c[4].metric("Data freshness", f"{qr.freshness_days} d")

    # ---- degradation logic ----
    drift = _feature_drift(a, bundle)
    max_drift = drift["abs_drift"].max()
    recent_probs = _recent_scores(scores)
    degraded, reasons = _degradation_flag(max_drift, qr, recent_probs, bundle)
    flag_color = charts.BAD if degraded else charts.GOOD
    st.markdown(
        f"<div class='card'><span class='metric-label'>Overall model health</span><br>"
        f"<span style='color:{flag_color};font-size:1.3rem;font-weight:700'>"
        f"{'⚠ REVIEW RECOMMENDED' if degraded else '✓ HEALTHY'}</span>"
        f"<br><span style='color:#8ea0bd'>{'; '.join(reasons) if reasons else 'All monitoring checks within tolerance.'}</span></div>",
        unsafe_allow_html=True)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Feature drift")
        st.caption("Standardised shift of recent (last 30 days) feature means vs "
                   "the training distribution. |drift| > 1 SD is worth a look.")
        st.plotly_chart(_drift_chart(drift), width='stretch')
    with right:
        st.subheader("Prediction distribution")
        st.caption("Distribution of current risk scores across the squad's recent "
                   "monitoring days. A collapse to 0/1 or a large shift signals trouble.")
        st.plotly_chart(charts.prediction_distribution_chart(recent_probs),
                        width='stretch')

    st.subheader("Data inputs")
    cc = st.columns(4)
    cc[0].metric("Completeness", f"{qr.overall_completeness:.1f}%")
    cc[1].metric("Missing (worst col)",
                 f"{max(qr.missing_by_column.values()) if qr.missing_by_column else 0}")
    cc[2].metric("Duplicates removed", qr.duplicate_records)
    cc[3].metric("Outliers capped", qr.outliers_capped)
    ui.disclaimer()


def _feature_drift(a, bundle):
    cols = bundle["feature_cols"]
    ref = bundle["drift_reference"]
    table = a["table"]
    last = pd.to_datetime(table["date"]).max()
    recent = table[pd.to_datetime(table["date"]) > last - pd.Timedelta(days=30)]
    rows = []
    for c in cols:
        cur = float(recent[c].astype(float).mean())
        mu = float(ref["means"].get(c, cur))
        sd = float(ref["stds"].get(c, 1.0)) or 1e-9
        drift = (cur - mu) / sd
        rows.append({"feature": c, "label": FEATURE_META.get(c, (c, True))[0],
                     "drift": drift, "abs_drift": abs(drift)})
    return pd.DataFrame(rows).sort_values("abs_drift", ascending=False)


def _drift_chart(drift):
    d = drift.head(12).iloc[::-1]
    colors = [charts.BAD if abs(v) > 1 else (charts.WARN if abs(v) > 0.5 else charts.PRIMARY)
              for v in d["drift"]]
    fig = go.Figure(go.Bar(x=d["drift"], y=d["label"], orientation="h",
                           marker_color=colors,
                           hovertemplate="%{y}<br>drift: %{x:+.2f} SD<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=charts.MUTED, width=1))
    fig.add_vline(x=1, line=dict(color=charts.BAD, width=1, dash="dot"))
    fig.add_vline(x=-1, line=dict(color=charts.BAD, width=1, dash="dot"))
    fig.update_xaxes(title="drift (SD vs training)")
    return charts.apply_theme(fig, height=420)


def _recent_scores(scores):
    last = pd.to_datetime(scores["date"]).max()
    recent = scores[pd.to_datetime(scores["date"]) > last - pd.Timedelta(days=14)]
    return recent["risk_probability"].to_numpy()


def _degradation_flag(max_drift, qr, recent_probs, bundle):
    reasons = []
    if max_drift > 1.5:
        reasons.append(f"large feature drift ({max_drift:.1f} SD)")
    if qr.overall_completeness < 90:
        reasons.append(f"data completeness low ({qr.overall_completeness:.0f}%)")
    if qr.freshness_days > 3:
        reasons.append(f"data stale ({qr.freshness_days} d)")
    if len(recent_probs) and (recent_probs.std() < 0.02):
        reasons.append("prediction distribution collapsed")
    return (len(reasons) > 0), reasons
