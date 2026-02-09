import streamlit as st

st.set_page_config(
    page_title="OncoRisk™",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.success("✅ THIS app.py is running")



# ============================================================
# OncoRisk – Dual Cox + RSF Clinical Survival Platform
# ============================================================

import os
from pathlib import Path
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def load_bg_image(path):
    path = Path(path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored


# ============================================================
# SESSION STATE (SINGLE SOURCE OF TRUTH)
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Overview"

PAGES = [
    "Overview",
    "Dashboard",
    "Patient Risk Profile Inference",
    "Cohort Analytics",
    "Feature Intelligence",
    "Gene Intelligence",
    "Model Facts",
]

# ============================================================
# URL → SESSION STATE SYNC (REQUIRED FOR CTA NAVIGATION)
# ============================================================
# ---- URL → session sync (CTA support) ----
if "page" in st.query_params:
    qp = st.query_params["page"]
    if qp in PAGES:
        st.session_state.page = qp


st.sidebar.title("OncoRisk Dual")
st.sidebar.markdown("Cox + RSF survival intelligence")
        
page = st.sidebar.selectbox(
    "Navigation",
    PAGES,
    index=PAGES.index(st.session_state.page),
)

st.session_state.page = page

st.write("🔎 PAGE VALUE =", repr(page))




# ============================================================
# GLOBAL STYLE
# ============================================================

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

st.markdown("""
<style>

/* ===== STREAMLIT HEADERS (st.header / st.subheader) ===== */
h1[data-testid="stHeader"] {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    margin-top: 3.5rem !important;
    margin-bottom: 1.5rem !important;
    color: #ffffff !important;
}

h2[data-testid="stHeader"] {
    font-size: 2.1rem !important;
    font-weight: 750 !important;
    margin-top: 3.0rem !important;
    margin-bottom: 1.3rem !important;
    color: #ffffff !important;
}

h3[data-testid="stHeader"] {
    font-size: 1.6rem !important;
    font-weight: 650 !important;
    margin-top: 2.2rem !important;
    margin-bottom: 1.0rem !important;
    color: #ffffff !important;
}
/* ===== HERO CTA BUTTON ===== */
.hero-cta {
    display: inline-block;
    margin-top: 2rem;
    padding: 0.9rem 1.6rem;
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    background: linear-gradient(
        135deg,
        #e5e7eb 0%,
        #ffffff 100%
    );
    border-radius: 999px;
    text-decoration: none;
    box-shadow: 
        0 10px 25px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.6);
    transition: all 0.25s ease;
}

.hero-cta:hover {
    transform: translateY(-2px);
    box-shadow:
        0 14px 30px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.7);
}

.hero-cta:active {
    transform: translateY(0);
}


/* Kill Streamlit spacing suppression */

/* ===== EXPANDER HEADERS ===== */
button[aria-expanded] {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* ===== BODY TEXT ===== */
/* ===== MAIN APP BACKGROUND & TEXT ===== */
[data-testid="stAppViewContainer"] {
    background-color: #111418;
    color: #e5e7eb;
}

/* Main content text */
.block-container {
    color: #e5e7eb;
}


/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #151a20;
    border-right: 1px solid #2b2f38;
}

/* Layout */
.block-container {
    padding: 1.5rem 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Force readable text everywhere */
body, p, div, span, li {
    color: #e5e7eb !important;
}

/* Titles */
h1, h2, h3, h4 {
    color: #f8fafc !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}
[data-testid="stMetricLabel"] {
    color: #cbd5f5 !important;
}

/* Captions */
[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "Models"
SPLITS_DIR = MODELS_DIR / "CoxPH_Final" / "splits"

CLINICOGENOMIC_DIR = MODELS_DIR / "Clinicogenomic_31genes_v2"
TCGA_PATH = CLINICOGENOMIC_DIR / "tables" / "tcga_clinicogenomic_31genes_with_surv.csv"
MB_PATH   = CLINICOGENOMIC_DIR / "tables" / "metabric_clinicogenomic_31genes_with_surv.csv"

# ============================================================
# SURVIVAL HELPERS
# ============================================================

def median_survival_time(times, surv):
    below = surv <= 0.5
    if not np.any(below):
        return np.nan
    return float(times[np.argmax(below)])

def rmst(times, surv):
    return float(np.trapz(surv, times))

def agreement_score(median_cox, median_rsf):
    if np.isnan(median_cox) or np.isnan(median_rsf):
        return np.nan
    denom = max(median_cox, median_rsf)
    if denom == 0:
        return np.nan
    delta = abs(median_cox - median_rsf) / denom
    return float(max(0.0, min(1.0, 1.0 - delta)))

def agreement_label(score):
    if np.isnan(score):
        return "Unknown"
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Moderate"
    return "Low"

