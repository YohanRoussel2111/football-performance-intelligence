"""Model Performance — comparison, temporal validation, metrics, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import app_core, ui
from src.visualization import charts


def render():
    ui.header("Model Performance", "temporal validation, metrics & calibration")
    bundle = app_core.get_bundle()
    meta = bundle["metadata"]
    comp = bundle["comparison"]
    test = bundle["test_metrics"]

    c = st.columns(5)
    c[0].metric("Selected model", meta["selected_model"])
    c[1].metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")
    c[2].metric("Test PR-AUC", f"{test['pr_auc']:.3f}")
    c[3].metric("Recall", f"{test['recall']:.2f}")
    c[4].metric("Precision", f"{test['precision']:.2f}")

    st.caption(f"Target: {meta['target_definition']}  ·  "
               f"Base rate (train) {meta['train_base_rate']:.1%}  ·  "
               f"n train/valid/test = {meta['n_train']}/{meta['n_valid']}/{meta['n_test']}")

    st.divider()
    st.subheader("Model comparison (validation window)")
    rows = []
    for name, m in comp.items():
        rows.append({
            "Model": name + (" ✓" if name == meta["selected_model"] else ""),
            "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
            "F1": m["f1"], "Recall": m["recall"], "Precision": m["precision"],
            "Temporal CV PR-AUC": m["temporal_cv_pr_auc"], "Brier": m["brier"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch',
                 column_config={col: st.column_config.NumberColumn(format="%.3f")
                                for col in ["PR-AUC", "ROC-AUC", "F1", "Recall",
                                            "Precision", "Temporal CV PR-AUC", "Brier"]})
    st.caption("Selection is by temporal cross-validation PR-AUC — the metric that "
               "matters most for rare events, estimated on expanding-window folds.")

    left, right = st.columns(2)
    with left:
        st.subheader("Calibration")
        st.plotly_chart(
            charts.calibration_chart(test["calibration_curve"], test["brier"]),
            width='stretch')
        st.caption(f"Calibration lowered the test Brier score from "
                   f"**{bundle['uncalibrated_test_brier']:.3f}** (raw) to "
                   f"**{test['brier']:.3f}** (calibrated). Calibrated probabilities "
                   "let staff read a 30% risk as *genuinely* ~30% of the time.")
    with right:
        st.subheader("Confusion matrix (test)")
        st.plotly_chart(_confusion(test["confusion_matrix"]), width='stretch')
        st.caption(f"At the operating threshold {test['threshold']:.2f}, tuned for "
                   "recall: a screening tool should rarely miss a genuine case, "
                   "accepting some false alarms that staff can quickly rule out.")

    st.divider()
    with st.expander("Why temporal validation (not a random split)?"):
        st.markdown(
            "In a longitudinal setting every prediction is made about the *future* "
            "from the *past*. A random train/test split leaks future information "
            "(and repeated measures of the same player) into training, inflating "
            "the score. We therefore split strictly by calendar time — earliest "
            f"{app_core.config.TRAIN_FRACTION:.0%} of the season trains, the next "
            f"{app_core.config.VALID_FRACTION:.0%} validates/calibrates, and the "
            "final window tests. The visible train→test drop is the honest cost of "
            "non-stationarity (fixture congestion shifts the base rate across the "
            "season) and is exactly what deployment would face.")


def _confusion(cm):
    cm = np.array(cm)
    labels = ["No event", "Event"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
        colorscale="Teal", showscale=False,
        text=cm, texttemplate="%{text}", textfont=dict(size=20, color="#e7ecf4"),
        hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
    return charts.apply_theme(fig, height=360)
