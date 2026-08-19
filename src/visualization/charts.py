"""
Plotly chart library and a single shared visual theme.

Everything reads as one system: one dark, sober professional palette, consistent
axes/gridlines, status colours that are colour-blind-aware (green/amber/red are
distinguished by luminance as well as hue), and tooltips on every metric. The
goal is a performance-department look — legible to a data scientist and to a
physio or coach alike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
BG = "#0d1420"
PANEL = "#131c2b"
GRID = "#26324a"
INK = "#e7ecf4"
MUTED = "#8ea0bd"

PRIMARY = "#3fb6c9"     # teal - load
ACCENT = "#7c9cff"      # periwinkle - secondary series
GOOD = "#2fb380"        # green
WARN = "#e0a93b"        # amber
BAD = "#e5484d"         # red
NEUTRAL = "#5a6b86"

STATUS_COLORS = {
    # availability
    "available": GOOD, "modified_training": WARN, "unavailable": BAD,
    "Available": GOOD, "Monitor": WARN, "Modified": WARN, "Unavailable": BAD,
    # monitoring levels
    "LOW": GOOD, "MODERATE": WARN, "HIGH": BAD,
    # dimensional
    "Normal": GOOD, "Elevated": WARN, "High": BAD,
    "Good": GOOD, "Moderate": WARN, "Poor": BAD,
    "Stable": GOOD, "Reduced": WARN, "Critical": BAD,
    "Low": GOOD,
    # pmi band
    "Green": GOOD, "Amber": WARN, "Red": BAD,
}

CATEGORICAL = [PRIMARY, ACCENT, WARN, GOOD, "#c58bd6", "#e08b6f", MUTED]


def apply_theme(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=INK, family="Inter, Segoe UI, sans-serif", size=13),
        margin=dict(l=50, r=24, t=48 if title else 20, b=40),
        height=height,
        # Always set an explicit (possibly empty) title to avoid plotly rendering
        # a stray "undefined" label on untitled charts.
        title=dict(text=title or "", font=dict(size=15, color=INK)),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor=PANEL, font_size=12),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# --------------------------------------------------------------------------- #
# Player charts
# --------------------------------------------------------------------------- #
def load_trend_chart(df: pd.DataFrame) -> go.Figure:
    """7-day vs 28-day load with match-day markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["load_7d"], name="Charge aiguë (7 j)",
        line=dict(color=PRIMARY, width=2.2), mode="lines",
        hovertemplate="%{x|%d %b}<br>Charge 7 j : %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["load_28d"] / 4, name="Charge chronique (28 j ÷4)",
        line=dict(color=ACCENT, width=1.8, dash="dot"), mode="lines",
        hovertemplate="%{x|%d %b}<br>Charge 28 j ÷4 : %{y:.0f}<extra></extra>"))
    matches = df[df["played_match"] == True] if "played_match" in df else df.iloc[0:0]
    if len(matches):
        fig.add_trace(go.Scatter(
            x=matches["date"], y=matches["load_7d"], name="Match",
            mode="markers", marker=dict(color=WARN, size=7, symbol="diamond"),
            hovertemplate="Match %{x|%d %b}<br>minutes : %{customdata}<extra></extra>",
            customdata=matches["minutes_played"] if "minutes_played" in matches else None))
    return apply_theme(fig, title="Charge d'entraînement & de match")


def acwr_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=0.8, y1=1.3, fillcolor=GOOD, opacity=0.10, line_width=0,
                  annotation_text="zone optimale", annotation_position="top left",
                  annotation_font_color=MUTED)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["acwr"], name="ACWR",
        line=dict(color=PRIMARY, width=2), mode="lines",
        hovertemplate="%{x|%d %b}<br>ACWR: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1.5, line=dict(color=BAD, width=1, dash="dash"))
    return apply_theme(fig, title="Ratio charge aiguë:chronique (ACWR)", height=300)


def wellness_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    series = [("fatigue", "Fatigue", WARN), ("muscle_soreness", "Courbatures", BAD),
              ("sleep_quality", "Qualité du sommeil", PRIMARY), ("stress", "Stress", ACCENT)]
    for col, name, color in series:
        if col in df:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[f"{col}_7d"] if f"{col}_7d" in df else df[col],
                name=name, line=dict(color=color, width=1.8), mode="lines",
                hovertemplate="%{x|%d %b}<br>" + name + ": %{y:.1f}<extra></extra>"))
    return apply_theme(fig, title="Bien-être (moyenne mobile 7 j, échelle 1–7)", height=300)


def cmj_chart(df: pd.DataFrame) -> go.Figure:
    meas = df[df["cmj_measured"] == 1] if "cmj_measured" in df else df.dropna(subset=["cmj_height"])
    fig = go.Figure()
    if "cmj_baseline" in df:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["cmj_baseline"], name="Référence individuelle",
            line=dict(color=MUTED, width=1.4, dash="dot"), mode="lines",
            hovertemplate="référence : %{y:.1f} cm<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=meas["date"], y=meas["cmj_height"], name="Hauteur CMJ",
        mode="lines+markers", line=dict(color=PRIMARY, width=2),
        marker=dict(size=6),
        hovertemplate="%{x|%d %b}<br>CMJ : %{y:.1f} cm<extra></extra>"))
    return apply_theme(fig, title="Détente verticale (CMJ) vs référence individuelle", height=300)


def risk_trajectory_chart(scored: pd.DataFrame, bands: dict | None = None) -> go.Figure:
    fig = go.Figure()
    if bands:
        fig.add_hrect(y0=bands["high"], y1=1.0, fillcolor=BAD, opacity=0.08, line_width=0)
        fig.add_hrect(y0=bands["moderate"], y1=bands["high"], fillcolor=WARN,
                      opacity=0.07, line_width=0)
    fig.add_trace(go.Scatter(
        x=scored["date"], y=scored["risk_probability"], name="Risque modèle",
        line=dict(color=PRIMARY, width=2), mode="lines", fill="tozeroy",
        fillcolor="rgba(63,182,201,0.12)",
        hovertemplate="%{x|%d %b}<br>risque : %{y:.0%}<extra></extra>"))
    return apply_theme(fig, title="Trajectoire du risque (modèle)", height=300)


def shap_contribution_chart(explanation: dict, top_n: int = 8) -> go.Figure:
    factors = sorted(explanation["all"], key=lambda d: abs(d["shap"]), reverse=True)[:top_n]
    factors = factors[::-1]
    labels = [f["label"] for f in factors]
    vals = [f["shap"] for f in factors]
    colors = [BAD if v > 0 else GOOD for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=colors,
        hovertemplate="%{y}<br>contribution : %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(xaxis_title="← diminue le risque    contribution au risque    augmente le risque →")
    return apply_theme(fig, title="Pourquoi ce joueur est-il signalé ? (SHAP)", height=340)


def calibration_chart(curve: dict, brier: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Calibration parfaite",
                             line=dict(color=MUTED, width=1.2, dash="dash"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=curve["mean_predicted"], y=curve["observed_frequency"],
        name="Modèle", mode="lines+markers", line=dict(color=PRIMARY, width=2),
        marker=dict(size=8),
        hovertemplate="prédit : %{x:.0%}<br>observé : %{y:.0%}<extra></extra>"))
    title = "Courbe de calibration (fiabilité)"
    if brier is not None:
        title += f"  ·  Brier = {brier:.3f}"
    fig.update_xaxes(title="Probabilité prédite", range=[0, 1])
    fig.update_yaxes(title="Fréquence observée", range=[0, 1])
    return apply_theme(fig, title=title, height=360)


def global_importance_chart(imp: pd.DataFrame, top_n: int = 12) -> go.Figure:
    d = imp.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["mean_abs_shap"], y=d["label"], orientation="h",
        marker_color=PRIMARY,
        hovertemplate="%{y}<br>|SHAP| moyen : %{x:.3f}<extra></extra>"))
    return apply_theme(fig, title="Importance globale des variables (|SHAP| moyen)", height=420)


def pmi_contributions_chart(contribs: dict) -> go.Figure:
    labels = list(contribs.keys())
    vals = list(contribs.values())
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=[PRIMARY, ACCENT, WARN, NEUTRAL],
        hovertemplate="%{x}<br>contribution : %{y:.1f}<extra></extra>"))
    fig.update_yaxes(title="Contribution à l'IPM")
    return apply_theme(fig, title="Contributions par dimension (IPM)", height=300)


def squad_load_bar(latest: pd.DataFrame) -> go.Figure:
    d = latest.sort_values("load_7d", ascending=True)
    fig = go.Figure(go.Bar(
        x=d["load_7d"], y=d["player_name"], orientation="h",
        marker_color=[STATUS_COLORS.get(s, NEUTRAL) for s in d.get("monitoring_level", [])] or PRIMARY,
        hovertemplate="%{y}<br>charge 7 j : %{x:.0f}<extra></extra>"))
    return apply_theme(fig, title="Charge de l'effectif (7 jours)", height=max(360, 18 * len(d)))


def prediction_distribution_chart(probs, ref_hist=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=probs, nbinsx=20, marker_color=PRIMARY,
                               opacity=0.85, name="Actuel"))
    fig.update_xaxes(title="Probabilité de risque prédite", range=[0, 1])
    fig.update_yaxes(title="Effectif")
    return apply_theme(fig, title="Distribution des prédictions", height=300)
