# main.py
import joblib

print("📦 Loading model...")

artifact = joblib.load("combined_model_v10.pkl")

print("✅ Keys in model:", artifact.keys())

print("Log model:", type(artifact["log_model"]))
print("XGB model:", type(artifact["xgb_model"]))
print("Features:", len(artifact["features"]))

print("🎉 Model is working correctly!")