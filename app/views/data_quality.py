"""Data Quality — completeness, missingness, duplicates, outliers, freshness."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import app_core, ui
from src.visualization import charts


def render():
    ui.header("Data Quality", "trust the inputs before trusting the model")
    a = app_core.get_analysis()
    qr = a["quality_report"]

    flag = qr.quality_flag
    color = {"OK": charts.GOOD, "WARNING": charts.WARN, "INSUFFICIENT": charts.BAD}[flag]
    st.markdown(
        f"<div class='card'><span class='metric-label'>Overall data quality</span><br>"
        f"<span style='color:{color};font-size:1.4rem;font-weight:700'>{flag}</span>"
        f"  ·  completeness {qr.overall_completeness:.1f}%</div>",
        unsafe_allow_html=True)
    if flag == "INSUFFICIENT":
        st.error("Data quality is insufficient — model outputs should not be relied "
                 "upon until inputs are corrected.")
    elif flag == "WARNING":
        st.warning("Data quality warnings present — interpret model outputs with care.")

    st.divider()
    c = st.columns(6)
    c[0].metric("Observations", f"{qr.n_observations:,}")
    c[1].metric("Players", qr.n_players)
    c[2].metric("Duplicates", qr.duplicate_records)
    c[3].metric("Outliers capped", qr.outliers_capped)
    c[4].metric("Last update", qr.last_update)
    c[5].metric("Freshness", f"{qr.freshness_days} d")

    st.caption(f"Coverage: {qr.date_min} → {qr.date_max}")

    left, right = st.columns(2)
    with left:
        st.subheader("Completeness by field")
        comp = pd.DataFrame({
            "field": list(qr.completeness_pct.keys()),
            "completeness": list(qr.completeness_pct.values()),
        }).sort_values("completeness")
        fig = go.Figure(go.Bar(
            x=comp["completeness"], y=comp["field"], orientation="h",
            marker_color=[charts.GOOD if v >= 95 else (charts.WARN if v >= 90 else charts.BAD)
                          for v in comp["completeness"]],
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_xaxes(range=[80, 100], title="% complete")
        st.plotly_chart(charts.apply_theme(fig, height=420), width='stretch')

    with right:
        st.subheader("Missing values by field")
        miss = pd.DataFrame({
            "Field": list(qr.missing_by_column.keys()),
            "Missing": list(qr.missing_by_column.values()),
        }).sort_values("Missing", ascending=False)
        st.dataframe(miss, hide_index=True, width='stretch', height=300)

        st.subheader("Outliers detected & capped")
        if qr.outliers_by_column:
            ob = pd.DataFrame({"Field": list(qr.outliers_by_column.keys()),
                               "Count": list(qr.outliers_by_column.values())})
            st.dataframe(ob, hide_index=True, width='stretch')
        else:
            st.success("No implausible values detected.")

    with st.expander("What the pipeline checks"):
        st.markdown(
            "- **Schema & labels** — required tables present, availability labels valid\n"
            "- **Duplicates** — identical player-day rows from double imports are removed\n"
            "- **Missing values** — wellness forms not completed, GPS units not worn\n"
            "- **Outliers** — values outside physiological plausibility bounds are "
            "flagged and capped (non-destructively), never silently dropped\n"
            "- **Unit harmonisation** — distances to metres, times to seconds\n"
            "- **Freshness** — days since the most recent observation")
    ui.disclaimer()
