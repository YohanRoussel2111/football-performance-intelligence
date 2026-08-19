"""Composants d'interface partagés : CSS global, en-tête de marque, pastilles de
statut, traductions d'affichage.

Toute l'identité visuelle du produit est portée par la feuille de style injectée
ci-dessous, qui restyle les widgets natifs de Streamlit (métriques, conteneurs,
onglets, boutons, tableaux) pour un rendu cohérent « département performance ».

Les valeurs internes des statuts restent en anglais (elles servent de clés dans
la logique analytique) ; seul l'AFFICHAGE est traduit en français via les
dictionnaires ci-dessous.
"""
from __future__ import annotations

import streamlit as st

from src.visualization.charts import GOOD, WARN, BAD, NEUTRAL, PRIMARY, ACCENT, INK, MUTED

STATUS_COLOR = {
    "available": GOOD, "modified_training": WARN, "unavailable": BAD,
    "LOW": GOOD, "MODERATE": WARN, "HIGH": BAD,
    "Normal": GOOD, "Elevated": WARN, "High": BAD,
    "Good": GOOD, "Moderate": WARN, "Poor": BAD,
    "Stable": GOOD, "Reduced": WARN, "Critical": BAD, "Low": GOOD,
    "Green": GOOD, "Amber": WARN, "Red": BAD, "OK": GOOD,
    "WARNING": WARN, "INSUFFICIENT": BAD,
    # variantes françaises
    "FAIBLE": GOOD, "MODÉRÉ": WARN, "ÉLEVÉ": BAD,
}

# --------------------------------------------------------------------------- #
# Traductions d'affichage (valeur interne -> libellé français)
# --------------------------------------------------------------------------- #
FR_LEVEL = {"LOW": "FAIBLE", "MODERATE": "MODÉRÉ", "HIGH": "ÉLEVÉ"}
FR_AVAIL = {"available": "Disponible", "modified_training": "Aménagé",
            "unavailable": "Indisponible"}
FR_LOAD = {"Normal": "Normale", "Elevated": "Élevée", "High": "Haute"}
FR_RECOV = {"Good": "Bonne", "Moderate": "Modérée", "Poor": "Faible"}
FR_NM = {"Stable": "Stable", "Reduced": "Réduit", "Critical": "Critique"}
FR_CONG = {"Low": "Faible", "Moderate": "Modérée", "High": "Élevée"}
FR_BAND = {"Green": "Vert", "Amber": "Orange", "Red": "Rouge"}
FR_QUALITY = {"OK": "BONNE", "WARNING": "À SURVEILLER", "INSUFFICIENT": "INSUFFISANTE"}

# Libellés génériques utilisés par chip()
LABELS = {
    **FR_AVAIL, **FR_LEVEL,
    **{k: v for k, v in FR_LOAD.items()},
    "Good": "Bonne", "Poor": "Faible",
    "Reduced": "Réduit", "Critical": "Critique",
    "Green": "Vert", "Amber": "Orange", "Red": "Rouge",
}

# emoji-préfixés pour les tableaux
def fr_avail_tag(v: str) -> str:
    return {"available": "🟢 Disponible", "modified_training": "🟡 Aménagé",
            "unavailable": "🔴 Indisponible"}.get(v, v)

def fr_load_tag(v: str) -> str:
    return {"Normal": "🟢 Normale", "Elevated": "🟡 Élevée", "High": "🔴 Haute"}.get(v, v)

def fr_recov_tag(v: str) -> str:
    return {"Good": "🟢 Bonne", "Moderate": "🟡 Modérée", "Poor": "🔴 Faible"}.get(v, v)

def fr_nm_tag(v: str) -> str:
    return {"Stable": "🟢 Stable", "Reduced": "🟡 Réduit", "Critical": "🔴 Critique"}.get(v, v)

def fr_level_tag(v: str) -> str:
    return {"LOW": "🟢 FAIBLE", "MODERATE": "🟡 MODÉRÉ", "HIGH": "🔴 ÉLEVÉ"}.get(v, v)


# Blason de marque — un écusson avec une courbe « données → performance ».
CREST_SVG = """
<svg width="44" height="48" viewBox="0 0 42 46" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="fpiG" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#3fb6c9"/><stop offset="1" stop-color="#7c9cff"/>
    </linearGradient>
  </defs>
  <path d="M21 1 L39 9 V25 C39 36 31 42 21 45 C11 42 3 36 3 25 V9 Z"
        fill="rgba(63,182,201,0.12)" stroke="url(#fpiG)" stroke-width="1.6"/>
  <polyline points="10,31 17,24 23,27 32,14" fill="none" stroke="url(#fpiG)"
        stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="32" cy="14" r="2.7" fill="#7c9cff"/>
  <circle cx="10" cy="31" r="2.2" fill="#3fb6c9"/>
</svg>
"""

# La feuille de style est une chaîne simple (pas d'f-string) pour garder les
# accolades CSS littérales.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, [data-testid="stSidebar"] {
  font-family: 'Inter','Segoe UI',system-ui,sans-serif;
}

.stApp {
  background:
    radial-gradient(1100px 560px at 12% -12%, rgba(63,182,201,0.10), transparent 60%),
    radial-gradient(1000px 520px at 105% -5%, rgba(124,156,255,0.09), transparent 55%),
    #0b1220;
}
.block-container { padding-top: 2.2rem; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0c1728 0%, #0a1120 100%);
  border-right: 1px solid #1c2740;
}
section[data-testid="stSidebar"] * { color: #cdd8ea; }
[data-testid="stSidebarNav"] ul { padding-top: 4px; }
[data-testid="stSidebarNav"] a { border-radius: 9px; margin: 1px 6px; padding: 6px 10px; }
[data-testid="stSidebarNav"] a:hover { background: rgba(63,182,201,0.10); }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: linear-gradient(90deg, rgba(63,182,201,0.22), rgba(124,156,255,0.12));
  box-shadow: inset 3px 0 0 #3fb6c9;
}

h1,h2,h3,h4,h5 { color:#f2f6fc; font-weight:700; letter-spacing:.2px; }
h3 { font-size: 1.12rem; }

[data-testid="stMarkdownContainer"] h3::before,
[data-testid="stHeadingWithActionElements"] h3::before {
  content:""; display:inline-block; width:7px; height:7px; border-radius:2px;
  margin-right:9px; vertical-align:middle;
  background: linear-gradient(135deg,#3fb6c9,#7c9cff);
}

.fpi-brand { display:flex; gap:12px; align-items:center;
  padding:4px 2px 14px 2px; border-bottom:1px solid #1c2740; margin-bottom:10px; }
.fpi-brand-title { font-weight:800; font-size:.94rem; line-height:1.08; letter-spacing:.6px;
  background:linear-gradient(90deg,#eaf1fb,#8fb9ff);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.fpi-brand-sub { color:#5f86c9; font-size:.58rem; letter-spacing:1.3px; margin-top:6px; font-weight:600; }

.fpi-header { margin: 0 0 18px 0; }
.fpi-eyebrow { color:#7c9cff; font-size:.70rem; font-weight:700; letter-spacing:2.4px;
  text-transform:uppercase; }
.fpi-title { color:#f4f8ff; font-size:1.75rem; font-weight:800; line-height:1.12; margin-top:3px; }
.fpi-accent { height:3px; width:66px; margin-top:11px; border-radius:3px;
  background:linear-gradient(90deg,#3fb6c9,#7c9cff); }

[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(21,31,49,0.92), rgba(14,21,35,0.92));
  border:1px solid #223049; border-top:2px solid #3fb6c9;
  border-radius:14px; padding:14px 16px 12px 16px;
  box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 10px 22px rgba(0,0,0,0.28);
}
[data-testid="stMetricValue"] { font-size:1.6rem; font-weight:800; color:#f5f9ff; }
[data-testid="stMetricLabel"] p { color:#8ea0bd; text-transform:uppercase;
  letter-spacing:.6px; font-size:.70rem; font-weight:600; }
[data-testid="stMetricDelta"] { font-weight:600; }

[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(180deg, rgba(20,30,47,0.78), rgba(13,20,33,0.78));
  border:1px solid #223049 !important; border-radius:16px;
  box-shadow: 0 12px 28px rgba(0,0,0,0.30); transition: border-color .15s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:#2e415f; }

.card { background: linear-gradient(180deg, rgba(20,30,47,0.9), rgba(13,20,33,0.9));
  border:1px solid #223049; border-radius:14px; padding:16px 18px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.28); }
.metric-label { color:#8ea0bd; font-size:.72rem; text-transform:uppercase; letter-spacing:.7px; font-weight:600; }

.chip { display:inline-block; padding:3px 12px; border-radius:999px; font-size:.72rem;
  font-weight:700; color:#0b1220; box-shadow:0 2px 10px rgba(0,0,0,0.28); }

.sci { background: linear-gradient(180deg, rgba(124,156,255,0.08), rgba(63,182,201,0.05));
  border:1px solid #2b3a58; border-left:3px solid #7c9cff; border-radius:10px;
  padding:12px 15px; margin:6px 0; color:#c7d4ea; font-size:.9rem; }

button[data-baseweb="tab"] { font-weight:600; color:#9fb0cc; }
button[data-baseweb="tab"][aria-selected="true"] { color:#f2f6fc; }
[data-baseweb="tab-highlight"] { background: linear-gradient(90deg,#3fb6c9,#7c9cff) !important; height:3px; }
[data-baseweb="tab-border"] { background:#1c2740 !important; }

.stButton>button {
  border-radius:10px; border:1px solid #2a3b58; font-weight:600; color:#e3ebf7;
  background:linear-gradient(180deg,#17253c,#111d31);
}
.stButton>button:hover { border-color:#3fb6c9; color:#ffffff;
  box-shadow:0 6px 16px rgba(63,182,201,0.18); }

[data-testid="stExpander"] { border:1px solid #223049; border-radius:12px;
  background:rgba(19,28,43,0.5); }
[data-testid="stExpander"] summary:hover { color:#3fb6c9; }

[data-testid="stDataFrame"], .stDataFrame { border:1px solid #223049; border-radius:12px; }
[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#1c2740; }

::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-thumb { background:#243349; border-radius:8px; }
::-webkit-scrollbar-thumb:hover { background:#2d3f5a; }
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = ""):
    st.markdown(
        f"""<div class="fpi-header">
        <div class="fpi-eyebrow">{subtitle}</div>
        <div class="fpi-title">{title}</div>
        <div class="fpi-accent"></div></div>""",
        unsafe_allow_html=True,
    )


def brand_sidebar():
    st.sidebar.markdown(
        f"""<div class="fpi-brand">{CREST_SVG}
        <div><div class="fpi-brand-title">FOOTBALL<br>PERFORMANCE<br>INTELLIGENCE</div>
        <div class="fpi-brand-sub">DES DONNÉES AUX DÉCISIONS DE PERFORMANCE</div></div></div>""",
        unsafe_allow_html=True,
    )


def chip(text: str, status: str | None = None) -> str:
    color = STATUS_COLOR.get(status or text, NEUTRAL)
    label = LABELS.get(text, text)
    return f'<span class="chip" style="background:{color};">{label}</span>'


def status_chip(text: str, status: str | None = None):
    st.markdown(chip(text, status), unsafe_allow_html=True)


def science_box(text: str):
    """Encadré 'base scientifique' réutilisable."""
    st.markdown(f"<div class='sci'>🔬 <b>Base scientifique</b> — {text}</div>",
                unsafe_allow_html=True)


def disclaimer():
    st.caption(
        "⚠️ Outil d'aide à la décision sur données synthétiques. Les sorties du "
        "modèle sont un signal d'alerte précoce **opérationnel** destiné à "
        "déclencher une revue par le staff — **pas** un diagnostic médical ni une "
        "prédiction de blessure. Corrélation ≠ causalité."
    )
