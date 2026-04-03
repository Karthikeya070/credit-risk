# src/woe_binning.py
import pandas as pd
import numpy as np

def calculate_woe(df, feature, target):

    grouped = df.groupby(feature)[target].agg(["count","sum"])

    grouped.columns = ["total","bad"]

    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["dist_good"] = grouped["good"] / grouped["good"].sum()
    grouped["dist_bad"] = grouped["bad"] / grouped["bad"].sum()

    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])

    grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]

    return grouped
