# extract_model_stats.py
import joblib, json, os, sys

# This helper reads the trained pipeline and exports scaler values the frontend
# can inspect without loading sklearn in the browser.
p = "vigil_pipeline.pkl"

# Fail early with the current folder shown, since this script depends on being
# run from the project root.
if not os.path.exists(p):
    print(f"❌ ERROR: {p} not found in current folder:", os.getcwd())
    sys.exit(1)

# The scaler is inside the saved sklearn pipeline beside the model.
bundle = joblib.load(p)
scaler = bundle["pipeline"].named_steps["scaler"]

# Means and standard deviations describe how each feature was normalized.
means = scaler.mean_.tolist()
scales = scaler.scale_.tolist()

# Feature names must stay in the same order as training and prediction.
features = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts",
    "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Bwd Pkt Len Max",
    "Fwd Pkt Len Mean", "Bwd Pkt Len Mean", "Flow Byts/s",
    "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Mean", "Bwd IAT Mean", "Fwd PSH Flags",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "FIN Flag Cnt",
    "SYN Flag Cnt", "RST Flag Cnt", "ACK Flag Cnt"
]

# Round the values so model_stats.json stays readable in the dashboard.
out = {
    "scaler_mean": [round(x, 8) for x in means],
    "scaler_std":  [round(x, 8) for x in scales],
    "features": features
}

# The server exposes this same file through /model_stats.json.
out_path = "model_stats.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"✅ Wrote {out_path} with {len(means)} means and stds.")
