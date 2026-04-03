import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# FEATURE BUILDER
# ==============================
class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["emp_length"] = (
            X["emp_length"]
            .astype(str)
            .str.replace("+ years", "", regex=False)
            .str.replace(" years", "", regex=False)
            .str.replace("< 1 year", "0.5", regex=False)
            .str.replace("n/a", "0", regex=False)
        )
        X["emp_length"] = pd.to_numeric(X["emp_length"], errors="coerce")

        X["term"] = pd.to_numeric(
            X["term"].astype(str).str.replace(" months", "", regex=False),
            errors="coerce"
        )

        X["int_rate"] = pd.to_numeric(
            X["int_rate"].astype(str).str.replace("%", "", regex=False),
            errors="coerce"
        )

        num_cols = [
            "annual_inc", "loan_amnt", "dti", "revol_util",
            "delinq_2yrs", "inq_last_6mths", "pub_rec",
            "open_acc", "total_acc"
        ]

        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Feature engineering
        X["log_income"] = np.log1p(X["annual_inc"])
        X["loan_to_income"] = X["loan_amnt"] / (X["annual_inc"] + 1)

        return X.replace([np.inf, -np.inf], 0).fillna(0)


# ==============================
# KS METRIC
# ==============================
def compute_ks(y_true, probs):
    return ks_2samp(probs[y_true == 1], probs[y_true == 0]).statistic


# ==============================
# TRAIN FUNCTION
# ==============================
def train_model(df, features, model_path="combined_model_v10.pkl"):
    df["is_bad"] = df["is_bad"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df["is_bad"],
        test_size=0.3,
        random_state=42,
        stratify=df["is_bad"]
    )

    categorical = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]
    numerical = [col for col in features if col not in categorical]

    prep_pipe = Pipeline([
        ("features", FeatureBuilder()),
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        ]))
    ])

    logger.info("Transforming data...")
    X_train_proc = prep_pipe.fit_transform(X_train)
    X_test_proc = prep_pipe.transform(X_test)

    # ==============================
    # MODELS
    # ==============================
    log_model = LogisticRegression(max_iter=2000, random_state=42)
    log_model.fit(X_train_proc, y_train)

    xgb_base = XGBClassifier(
        objective='binary:logistic',
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )

    xgb_calibrated = CalibratedClassifierCV(
        xgb_base,
        method="isotonic",
        cv=3
    )
    xgb_calibrated.fit(X_train_proc, y_train)

    # ==============================
    # PROBABILITIES (ENSEMBLE)
    # ==============================
    train_probs = (
        0.6 * xgb_calibrated.predict_proba(X_train_proc)[:, 1] +
        0.4 * log_model.predict_proba(X_train_proc)[:, 1]
    )

    test_probs = (
        0.6 * xgb_calibrated.predict_proba(X_test_proc)[:, 1] +
        0.4 * log_model.predict_proba(X_test_proc)[:, 1]
    )

    # ==============================
    # METRICS
    # ==============================
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    ks = compute_ks(y_test.values, test_probs)

    logger.info(f"Train AUC: {train_auc:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"KS Score: {ks:.4f}")

    metrics = {
        "train_auc": float(train_auc),
        "test_auc": float(test_auc),
        "ks": float(ks)
    }

    # ==============================
    # SAVE EVERYTHING
    # ==============================
    joblib.dump({
        "log_model": Pipeline([("prep", prep_pipe), ("clf", log_model)]),
        "xgb_model": Pipeline([("prep", prep_pipe), ("clf", xgb_calibrated)]),
        "features": features,
        "metrics": metrics
    }, model_path)

    logger.info("Model + metrics saved successfully!")