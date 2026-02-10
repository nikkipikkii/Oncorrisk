import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from graph import get_hero_graphs
from gene import get_gene_intelligence
# ============================================================
# DIRECT IMPORT FROM app.py (SOURCE OF TRUTH)
# ============================================================
# This imports the models, scalers, and the exact math logic
# used for Medians, RMST, and Agreement.
from app import (
    get_artifacts, 
    median_survival_time, 
    rmst, 
    agreement_score, 
    agreement_label
)

app = FastAPI(title="OncoRisk Dual-Model API")

# Enable CORS for React Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize artifacts globally
art = get_artifacts()
scaler = art["scaler"]
cph = art["cph"]
rsf = art["rsf"]
features = art["features"]
gene_features = art["gene_features"]

class InferenceRequest(BaseModel):
    age: float
    nodeStatus: str
    genes: Dict[str, float]

@app.get("/metadata")
async def get_metadata():
    """Syncs the React dropdowns with actual TCGA test patient IDs and genes."""
    df_tcga = art["df_tcga"]
    test_ids = art["df_test"].index.tolist()
    
    patients = []
    for pid in test_ids:
        patients.append({
            "id": pid,
            "age": float(df_tcga.loc[pid, "AGE"]),
            "node": 1 if df_tcga.loc[pid, "NODE_POS"] == 1 else 0
        })
    
    return {
        "genes": gene_features,
        "patients": patients
    }

@app.post("/predict")
async def predict(data: InferenceRequest):
    try:
        # 1. Feature Prep (Matches app.py logic)
        node_pos = 1 if data.nodeStatus == "Positive" else 0
        row_dict = {"AGE": data.age, "NODE_POS": node_pos}
        row_dict.update(data.genes)
        
        # Ensure all features are present in correct order
        row_raw = pd.DataFrame([row_dict])[features]
        X_row = scaler.transform(row_raw.values)
        df_row_scaled = pd.DataFrame(X_row, columns=features)

        # 2. Model Inference
        # Cox Hazard
        hazard_cox = float(cph.predict_partial_hazard(df_row_scaled).values[0])
        surv_func_cox = cph.predict_survival_function(df_row_scaled)
        times_cox = surv_func_cox.index.values.astype(float)
        surv_cox = surv_func_cox.values[:, 0].astype(float)

        # RSF Risk
        surv_funcs_rsf = rsf.predict_survival_function(X_row)
        sf_rsf = surv_funcs_rsf[0]
        times_rsf = sf_rsf.x.astype(float)
        surv_rsf = sf_rsf.y.astype(float)

        # 3. CRITICAL GRID SYNC (Fixes discrepancies in Median/RMST)
        t_min = 0.0
        t_max = min(times_cox.max(), times_rsf.max())
        
        # Matches app.py density (200 points) to eliminate interpolation jitter
        grid = np.linspace(t_min, t_max, 200) 

        surv_cox_grid = np.interp(grid, times_cox, surv_cox)
        surv_rsf_grid = np.interp(grid, times_rsf, surv_rsf)

        # 4. Metric Extraction (Direct logic reuse)
        m_cox = median_survival_time(grid, surv_cox_grid)
        m_rsf = median_survival_time(grid, surv_rsf_grid)
        
        r_cox = rmst(grid, surv_cox_grid)
        r_rsf = rmst(grid, surv_rsf_grid)
        
        consensus_median = np.nanmean([m_cox, m_rsf])
        agree = agreement_score(m_cox, m_rsf)

        # Logging for manual verification in Terminal
        print(f"\n--- Inference for Age: {data.age}, Node: {data.nodeStatus} ---")
        print(f"RSF Median: {m_rsf:.1f} | RSF RMST: {r_rsf:.1f}")

        return {
            "summary": {
                "coxHazard": round(hazard_cox, 2),
                "rsfRisk": round(float(sf_rsf.y[-1]), 2),
                "agreement": round(agree, 2) if not np.isnan(agree) else "N/A",
                "agreementLabel": agreement_label(agree)
            },
            "estimates": {
                "medianCox": round(m_cox, 1) if not np.isnan(m_cox) else None,
                "medianRsf": round(m_rsf, 1) if not np.isnan(m_rsf) else None,
                "consensus": round(consensus_median, 1) if not np.isnan(consensus_median) else None
            },
            "rmst": {
                "cox": round(r_cox, 1),
                "rsf": round(r_rsf, 1)
            },
            "curveData": [
                {"time": float(t), "cox": float(c), "rsf": float(r)} 
                for t, c, r in zip(grid, surv_cox_grid, surv_rsf_grid)
            ]
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# Graph data endpoint (for Hero visualizations)


@app.get("/hero-charts")
async def hero_charts():
    # 'art' is the artifacts dictionary loaded in your main.py
    return get_hero_graphs(art)
# ========

# ROUTE: GENE INTELLIGENCE
# =================================================
@app.get("/gene-intelligence")
async def gene_intelligence_endpoint():
    """
    Returns the ranked list of genes, coefficients, and narratives 
    derived directly from the model artifacts.
    """
    # 'art' is the global artifacts dictionary loaded on startup
    return get_gene_intelligence(art)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)