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

# ============================================================
# DATA + MODEL LOADING
# ============================================================

@st.cache_resource
def load_artifacts():
    df_tcga = pd.read_csv(TCGA_PATH, index_col=0)
    df_mb   = pd.read_csv(MB_PATH, index_col=0)

    clinical = ["AGE", "NODE_POS"]
    
    gene_features = [
        c for c in df_tcga.columns
        if c not in ["time", "event", "AGE", "NODE_POS"]
    ]
    features = clinical + gene_features

    train_ids = pd.read_csv(SPLITS_DIR / "train_ids.csv").iloc[:, 0]
    test_ids  = pd.read_csv(SPLITS_DIR / "test_ids.csv").iloc[:, 0]

    train_ids = train_ids[train_ids.isin(df_tcga.index)]
    test_ids  = test_ids[test_ids.isin(df_tcga.index)]

    df_train = df_tcga.loc[train_ids]
    df_test  = df_tcga.loc[test_ids]

    scaler = StandardScaler().fit(df_train[features])

    X_train = scaler.transform(df_train[features])
    X_test  = scaler.transform(df_test[features])
    X_mb    = scaler.transform(df_mb[features])

    df_train_s = pd.concat(
        [df_train[["time","event"]],
         pd.DataFrame(X_train, index=df_train.index, columns=features)], axis=1)

    df_test_s = pd.concat(
        [df_test[["time","event"]],
         pd.DataFrame(X_test, index=df_test.index, columns=features)], axis=1)

    df_mb_s = pd.concat(
        [df_mb[["time","event"]],
         pd.DataFrame(X_mb, index=df_mb.index, columns=features)], axis=1)

    # Cox
    cph = CoxPHFitter()
    cph.fit(df_train_s, "time", "event")

    risk_train_cox = cph.predict_partial_hazard(df_train_s).values.flatten()
    risk_test_cox  = cph.predict_partial_hazard(df_test_s).values.flatten()
    risk_mb_cox    = cph.predict_partial_hazard(df_mb_s).values.flatten()
    cindex_train_cox = concordance_index(
        df_train_s["time"],
        -risk_train_cox,
        df_train_s["event"]
    )


    cindex_test_cox  = concordance_index(df_test_s["time"],  -risk_test_cox,  df_test_s["event"])
    cindex_mb_cox    = concordance_index(df_mb_s["time"],    -risk_mb_cox,    df_mb_s["event"])

    # RSF
    rsf = RandomSurvivalForest(
        n_estimators=500,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=123,
        n_jobs=-1
    )
    rsf.fit(X_train, Surv.from_arrays(df_train["event"].astype(bool),
                                      df_train["time"].astype(float)))

    ch_train = rsf.predict_cumulative_hazard_function(X_train)
    ch_test  = rsf.predict_cumulative_hazard_function(X_test)
    ch_mb    = rsf.predict_cumulative_hazard_function(X_mb)

    risk_train_rsf = np.array([fn.y[-1] for fn in ch_train])
    risk_test_rsf  = np.array([fn.y[-1] for fn in ch_test])
    risk_mb_rsf    = np.array([fn.y[-1] for fn in ch_mb])


    cindex_train_rsf = concordance_index_censored(
        df_train["event"].astype(bool),
        df_train["time"].astype(float),
        risk_train_rsf)[0]

    cindex_test_rsf = concordance_index_censored(
        df_test["event"].astype(bool),
        df_test["time"].astype(float),
        risk_test_rsf)[0]

    cindex_mb_rsf = concordance_index_censored(
        df_mb["event"].astype(bool),
        df_mb["time"].astype(float),
        risk_mb_rsf)[0]

    # Feature importance
    df_imp_cox = cph.params_.to_frame("coef")
    df_imp_cox["feature"] = df_imp_cox.index
    df_imp_cox["abs_coef"] = df_imp_cox["coef"].abs()
    df_imp_cox["type"] = ["clinical" if f in clinical else "gene" for f in df_imp_cox.index]
    df_imp_cox = df_imp_cox.sort_values("abs_coef", ascending=False)


    try:
        df_imp_rsf = pd.DataFrame({
            "feature": features,
            "importance": rsf.feature_importances_,
        })
        df_imp_rsf["abs_importance"] = df_imp_rsf["importance"].abs()
        df_imp_rsf["type"] = ["clinical" if f in clinical else "gene" for f in features]
        df_imp_rsf = df_imp_rsf.sort_values("abs_importance", ascending=False)
    except Exception:
        df_imp_rsf = pd.DataFrame()

    # KM prep
    df_tcga_test_km = df_test_s.copy()
    df_tcga_test_km["risk_cox"] = risk_test_cox
    cut_tcga = np.median(risk_test_cox)
    df_tcga_test_km["group_cox"] = np.where(df_tcga_test_km["risk_cox"] > cut_tcga, "High", "Low")

    df_mb_km = df_mb_s.copy()
    df_mb_km["risk_cox"] = risk_mb_cox
    cut_mb = np.median(risk_mb_cox)
    df_mb_km["group_cox"] = np.where(df_mb_km["risk_cox"] > cut_mb, "High", "Low")

    return dict(
        df_tcga=df_tcga,
        df_mb=df_mb,
        df_test=df_test_s,
        df_mb_s=df_mb_s,
        features=features,
        gene_features=gene_features,
        scaler=scaler,
        cph=cph,
        rsf=rsf,
        risk_test_cox=risk_test_cox,
        risk_test_rsf=risk_test_rsf,
        cindex_train_cox=cindex_train_cox,
        cindex_train_rsf=cindex_train_rsf,
        df_tcga_test_km=df_tcga_test_km,
        df_mb_km=df_mb_km,
        df_imp_cox=df_imp_cox,
        df_imp_rsf=df_imp_rsf,
        cindex_test_cox=cindex_test_cox,
        cindex_test_rsf=cindex_test_rsf,
        cindex_mb_cox=cindex_mb_cox,
        cindex_mb_rsf=cindex_mb_rsf
    )
@st.cache_resource(show_spinner="Loading survival models...")
def get_artifacts():
    return load_artifacts()


# ============================================================
# GENE NARRATIVES
# ============================================================

GENE_NARRATIVES = {
    "CD24": "Immune evasion via Siglec-10; higher expression aligns with increased hazard.",
    "SLC16A2": "Thyroid hormone transport; metabolic rewiring under stress.",
    "SERPINA1": "Protease regulation and invasion-permissive microenvironment.",
    "TFPI2": "Extracellular matrix control and metastatic potential.",
    "SEMA3B": "Loss of repulsive anti-invasive cues.",
    "TNFRSF14": "Tumor–immune interface regulation.",
    "APOOL": "Mitochondrial lipid handling.",
    "MRPL13": "Oxidative phosphorylation dependency.",
    "QPRT": "NAD+ biosynthesis under stress.",
    "JCHAIN": "Humoral immune architecture.",
    "NANOS1": "Stem-like, therapy-tolerant state.",
    "EDA2R": "Stress-adaptive signaling.",
}
