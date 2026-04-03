# src/feature_engineering.py
def apply_woe(df, bins):

    for feature in bins:

        mapping = bins[feature]["woe"]

        df[feature+"_woe"] = df[feature].map(mapping)

    return df
