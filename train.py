# train.py
import pandas as pd
from src.train_scorecard import train_model

print("📂 Loading data...")

df = pd.read_csv("Accepted.csv", low_memory=False)

print("✅ Data loaded:", df.shape)

# ==============================
# TARGET CREATION
# ==============================
df["loan_status"] = df["loan_status"].astype(str)

bad_status = ["Charged Off", "Default"]

df = df[df["loan_status"].isin([
    "Fully Paid",
    "Charged Off",
    "Default"
])]

df["is_bad"] = df["loan_status"].apply(
    lambda x: 1 if x in bad_status else 0
)

# ==============================
# FEATURES
# ==============================
features = [
    "loan_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_util", "total_acc"
]

# ==============================
# TRAIN MODEL
# ==============================
print("🚀 Training model...")

train_model(
    df=df,
    features=features,
    model_path="combined_model_v10.pkl"
)

print("🎉 Training complete!")