"""Shared UI helpers: global CSS, header, status chips, metric cards."""
from __future__ import annotations

import streamlit as st

from src.visualization.charts import GOOD, WARN, BAD, NEUTRAL, PRIMARY, INK, MUTED

STATUS_COLOR = {
    "available": GOOD, "modified_training": WARN, "unavailable": BAD,
    "LOW": GOOD, "MODERATE": WARN, "HIGH": BAD,
    "Normal": GOOD, "Elevated": WARN, "High": BAD,
    "Good": GOOD, "Moderate": WARN, "Poor": BAD,
    "Stable": GOOD, "Reduced": WARN, "Critical": BAD, "Low": GOOD,
    "Green": GOOD, "Amber": WARN, "Red": BAD, "OK": GOOD,
    "WARNING": WARN, "INSUFFICIENT": BAD,
}

LABELS = {"available": "Available", "modified_training": "Modified",
          "unavailable": "Unavailable"}


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background: #0d1420; }}
        section[data-testid="stSidebar"] {{ background: #0a111c; }}
        h1, h2, h3, h4 {{ color: {INK}; font-family: 'Inter','Segoe UI',sans-serif; }}
        .fpi-title {{ font-size: 1.35rem; font-weight: 700; letter-spacing:.3px;
                     color:{INK}; margin-bottom:0; }}
        .fpi-sub {{ color:{MUTED}; font-size:.8rem; margin-top:-2px;
                    text-transform:uppercase; letter-spacing:1.5px; }}
        .chip {{ display:inline-block; padding:2px 10px; border-radius:11px;
                 font-size:.74rem; font-weight:600; color:#0d1420; }}
        .card {{ background:#131c2b; border:1px solid #26324a; border-radius:12px;
                 padding:14px 16px; }}
        .metric-big {{ font-size:1.8rem; font-weight:700; color:{INK}; }}
        .metric-label {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase;
                         letter-spacing:.6px; }}
        div[data-testid="stMetricValue"] {{ font-size:1.7rem; }}
        .stDataFrame {{ border:1px solid #26324a; border-radius:8px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str = ""):
    st.markdown(
        f"""<div style="display:flex;align-items:baseline;gap:14px;
        border-bottom:1px solid #26324a;padding-bottom:10px;margin-bottom:14px;">
        <span class="fpi-title">{title}</span>
        <span class="fpi-sub">{subtitle}</span></div>""",
        unsafe_allow_html=True,
    )


def brand_sidebar():
    st.sidebar.markdown(
        f"""<div style="padding:6px 4px 14px 4px;">
        <div style="font-weight:800;color:{INK};font-size:1.02rem;line-height:1.15;">
        FOOTBALL PERFORMANCE<br>INTELLIGENCE</div>
        <div style="color:{PRIMARY};font-size:.68rem;letter-spacing:1px;
        margin-top:4px;">FROM DATA TO ACTIONABLE DECISIONS</div></div>""",
        unsafe_allow_html=True,
    )


def chip(text: str, status: str | None = None) -> str:
    color = STATUS_COLOR.get(status or text, NEUTRAL)
    label = LABELS.get(text, text)
    return f'<span class="chip" style="background:{color};">{label}</span>'


def status_chip(text: str, status: str | None = None):
    st.markdown(chip(text, status), unsafe_allow_html=True)


def disclaimer():
    st.caption(
        "⚠️ Decision-support tool on synthetic data. Model outputs are an "
        "operational early-warning signal to prompt staff review — **not** a "
        "medical diagnosis or an injury prediction. Correlation ≠ causation."
    )
