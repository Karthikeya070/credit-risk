from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import numpy as np
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk API")

# ==============================
# SCORECARD CONFIG
# ==============================
PDO = 50
BASE_SCORE = 650
BASE_ODDS = 20

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

def prob_to_score(prob):
    prob = np.clip(prob, 0.0001, 0.9999)
    return int(OFFSET + FACTOR * np.log((1 - prob) / prob))

def get_decision(score):
    if score >= 620:
        return "APPROVE"
    elif score >= 580:
        return "REVIEW"
    else:
        return "REJECT"

# ✅ Risk Tier logic
def get_risk_tier(score):
    if score >= 700:
        return "A"
    elif score >= 650:
        return "B"
    elif score >= 600:
        return "C"
    elif score >= 550:
        return "D"
    else:
        return "E"

# ==============================
# LOAD MODEL
# ==============================
try:
    artifact = joblib.load("combined_model_v10.pkl")

    PREPROCESSOR = artifact["log_model"].named_steps["prep"]
    LOG_MODEL = artifact["log_model"].named_steps["clf"]
    XGB_WRAPPER = artifact["xgb_model"].named_steps["clf"]

    calibrated_model = XGB_WRAPPER.calibrated_classifiers_[0]
    XGB_CORE = calibrated_model.estimator

    if hasattr(calibrated_model, "calibrator"):
        CALIBRATOR = calibrated_model.calibrator
    else:
        CALIBRATOR = calibrated_model.calibrators[0]

    explainer = shap.TreeExplainer(XGB_CORE)

    METRICS = artifact.get("metrics", {})

    logger.info("Model + metrics loaded successfully")

except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

# ==============================
# INPUT SCHEMA
# ==============================
class CreditInput(BaseModel):
    loan_amnt: float
    int_rate: float
    term: str
    installment: float
    grade: str
    sub_grade: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    dti: float
    delinq_2yrs: int
    inq_last_6mths: int
    open_acc: int
    pub_rec: int
    revol_util: float
    total_acc: int

# ==============================
# API ENDPOINT
# ==============================
@app.post("/score")
async def score(data: CreditInput):
    try:
        df = pd.DataFrame([data.model_dump()])

        # ── Preprocess ──
        X = PREPROCESSOR.transform(df)
        X = np.asarray(X).astype(np.float64)

        # ── Predictions ──
        p_log = LOG_MODEL.predict_proba(X)[:, 1][0]

        raw_xgb = XGB_CORE.predict_proba(X)[:, 1]
        p_xgb = CALIBRATOR.transform(raw_xgb)[0]

        # ── Ensemble ──
        prob = (0.4 * p_log) + (0.6 * p_xgb)

        # ── Score + Decision ──
        score = prob_to_score(prob)
        decision = get_decision(score)
        risk_tier = get_risk_tier(score)

        # ── SHAP Explanation ──
        drivers = []
        try:
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
            else:
                shap_vals = shap_values[0]

            feature_names = PREPROCESSOR.named_steps["preprocess"].get_feature_names_out()
            impacts = dict(zip(feature_names, shap_vals))

            top_features = sorted(
                impacts.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            drivers = [
                {
                    "feature": f.replace("num__", "").replace("cat__", "").replace("_", " ").title(),
                    "impact": round(float(v), 4)
                }
                for f, v in top_features
            ]

        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

        # ── RESPONSE ──
        return {
            "score": int(score),
            "probability_of_default": round(float(prob), 4),
            "decision": decision,
            "risk_tier": risk_tier,   # ✅ FIXED
            "risk_drivers": drivers,
            "model_metrics": METRICS
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")
