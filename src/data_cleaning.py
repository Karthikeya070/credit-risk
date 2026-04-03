# src/data_cleaning.py
import pandas as pd
import numpy as np

def load_data(path):

    df = pd.read_csv(path, low_memory=False)

    bad = ["Charged Off","Default"]
    good = ["Fully Paid","Current"]

    df["is_bad"] = df["loan_status"].apply(
        lambda x: 1 if x in bad else (0 if x in good else None)
    )

    df = df.dropna(subset=["is_bad"])

    df["int_rate"] = df["int_rate"].astype(str).str.replace("%","").astype(float)

    df["revol_util"] = df["revol_util"].astype(str).str.replace("%","").astype(float)

    df["annual_inc"] = df["annual_inc"].astype(str).str.replace(",","").astype(float)

    df["log_income"] = np.log1p(df["annual_inc"])

    df = df.fillna(0)

    return df
