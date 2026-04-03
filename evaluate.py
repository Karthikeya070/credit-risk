import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

# ==============================
# LOAD MODEL
# ==============================
artifact = joblib.load("combined_model_v10.pkl")

PREPROCESSOR = artifact["log_model"].named_steps["prep"]
LOG_MODEL = artifact["log_model"].named_steps["clf"]
XGB_WRAPPER = artifact["xgb_model"].named_steps["clf"]

# 🔥 SAME EXTRACTION AS FASTAPI
calibrated_model = XGB_WRAPPER.calibrated_classifiers_[0]
XGB_CORE = calibrated_model.estimator

if hasattr(calibrated_model, "calibrator"):
    CALIBRATOR = calibrated_model.calibrator
else:
    CALIBRATOR = calibrated_model.calibrators[0]

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("Accepted.csv", low_memory=False)

bad_status = ["Charged Off", "Default"]

df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])]

df["is_bad"] = df["loan_status"].apply(lambda x: 1 if x in bad_status else 0)

features = artifact["features"]

X = df[features]
y = df["is_bad"]

# ==============================
# PREPROCESS
# ==============================
X_proc = PREPROCESSOR.transform(X)
X_proc = np.asarray(X_proc).astype(np.float64)

# ==============================
# PREDICTIONS (FIXED)
# ==============================
p_log = LOG_MODEL.predict_proba(X_proc)[:, 1]

raw_xgb = XGB_CORE.predict_proba(X_proc)[:, 1]
p_xgb = CALIBRATOR.transform(raw_xgb)

# Ensemble
probs = 0.4 * p_log + 0.6 * p_xgb

# ==============================
# METRICS
# ==============================
auc = roc_auc_score(y, probs)
ks = ks_2samp(probs[y == 1], probs[y == 0]).statistic

print(f"✅ AUC: {auc:.4f}")
print(f"✅ KS: {ks:.4f}")