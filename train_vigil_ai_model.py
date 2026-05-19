# train_vigil_ai_model.py (32-feature | XGBoost + MLP comparison)
import os, joblib, json, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
print("🚀 Starting Vigil AI dual-model training (XGBoost + MLP)...")

# ─── Load all .parquet files ───────────────────────────────────────────────────
# The CICDDoS dataset is split across attack files, so training starts by
# combining every parquet file into one shared dataframe.
base_path = os.path.join(os.path.dirname(__file__), "cicddos2019")
dfs = []
for root, _, files in os.walk(base_path):
    for f in files:
        if f.lower().endswith(".parquet"):
            try:
                dfs.append(pd.read_parquet(os.path.join(root, f), engine="pyarrow"))
                print(f"  ✔ Loaded: {f}")
            except Exception as e:
                print(f"  ⚠️  Skipped: {f} | {e}")

if not dfs:
    raise RuntimeError("❌ No .parquet files found in cicddos2019/")

df = pd.concat(dfs, ignore_index=True)
print(f"✅ Combined dataset shape: {df.shape}")

# ─── Label column ─────────────────────────────────────────────────────────────
# Some exported datasets use slightly different names, so I normalize the target
# column before the rest of the script assumes "Label" exists.
if "Label" not in df.columns:
    for cand in ["label", "Attack", "Category"]:
        if cand in df.columns:
            df.rename(columns={cand: "Label"}, inplace=True)
            break

# ─── Normalize label names ────────────────────────────────────────────────────
# The raw dataset has many specific attack names. The app groups them into the
# seven categories that appear in the dashboard and chatbot.
label_series = df["Label"].astype(str)
replacements = {
    "DrDoS": "DDoS", "WebDDoS": "DDoS", "UDP-lag": "DDoS", "UDPLag": "DDoS",
    "LDAP": "DDoS", "MSSQL": "DDoS", "NetBIOS": "DDoS", "NTP": "DDoS",
    "TFTP": "DDoS", "Syn": "DoS", "UDP": "DoS", "Portmap": "PortScan"
}
for old, new in replacements.items():
    label_series = label_series.str.replace(old, new, regex=False)

id_map = {
    "Benign": 0, "DDoS": 1, "DoS": 2, "PortScan": 3,
    "Botnet": 4, "Infiltration": 5, "WebAttack": 6
}
id_to_label = {
    0: "Benign (Normal Traffic)", 1: "DDoS Attack", 2: "DoS Attack",
    3: "Port Scan Activity",      4: "Botnet / Malware Behavior",
    5: "Infiltration Attempt",    6: "Web Attack Detected"
}

y = label_series.map(lambda x: next((k for k in id_map if k.lower() in x.lower()), None))
# Unknown labels default to Benign so a rare dataset typo does not stop the
# whole training run. It is conservative for demo stability, not ideal research.
y = y.fillna("Benign").map(id_map).astype(int)

# ─── 32 features ──────────────────────────────────────────────────────────────
# These are the same 32 numeric inputs exposed in the web form, so the trained
# pipeline and frontend stay aligned.
selected_features = [
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

numeric_df = df.select_dtypes(include=[np.number])
missing = [c for c in selected_features if c not in numeric_df.columns]
if missing:
    print(f"  ⚠️  Missing columns (filled with 0): {missing}")

# reindex keeps the feature order fixed, which matters because sklearn treats
# the array position as the meaning of each value.
X = numeric_df.reindex(columns=selected_features, fill_value=0)
print(f"📊 Feature shape: {X.shape}")
print(f"🧩 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ─── SMOTE + split ────────────────────────────────────────────────────────────
print("\n⚖️  Applying SMOTE oversampling...")
# Attack classes are uneven, so SMOTE gives the smaller classes enough examples
# for the model comparison to be meaningful in the prototype.
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ─── Helper: compute per-class F1 ─────────────────────────────────────────────
def per_class_f1(y_true, y_pred, id_to_label):
    # Per-class F1 is easier to explain on the dashboard than a full text report.
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    result = {}
    for class_id, label_name in id_to_label.items():
        key = str(class_id)
        if key in report:
            result[label_name] = round(report[key]["f1-score"] * 100, 2)
    return result

# ─── Helper: compute summary metrics ─────────────────────────────────────────
def compute_metrics(y_true, y_pred, elapsed):
    # Weighted averages keep the overall score fair after balancing multiple
    # attack categories.
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred,    average="weighted", zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_true, y_pred,        average="weighted", zero_division=0) * 100, 2),
        "train_time_sec": round(elapsed, 1),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 1.  XGBoost  (production model)
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━ [1/2] Training XGBoost (production model) ━━━")
scaler_xgb = StandardScaler()
# XGBoost is the production pick because it trains quickly and usually handles
# tabular network-flow features better than a small neural network.
xgb_model  = XGBClassifier(
    n_estimators=250, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric="mlogloss", random_state=42
)
xgb_pipe = Pipeline([("scaler", scaler_xgb), ("model", xgb_model)])

t0 = time.time()
xgb_pipe.fit(X_train, y_train)
xgb_elapsed = time.time() - t0

y_pred_xgb   = xgb_pipe.predict(X_test)
xgb_metrics  = compute_metrics(y_test, y_pred_xgb, xgb_elapsed)
xgb_per_class = per_class_f1(y_test, y_pred_xgb, id_to_label)

print(f"   ✅ XGBoost done in {xgb_elapsed:.1f}s | Accuracy: {xgb_metrics['accuracy']}%")
print(classification_report(y_test, y_pred_xgb, zero_division=0))

# Save XGBoost as the main production pipeline used by model.py and /predict.
joblib.dump({"pipeline": xgb_pipe, "id_to_label": id_to_label}, "vigil_pipeline.pkl")
print("   💾 Saved vigil_pipeline.pkl (XGBoost production model)")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MLP  (deep learning comparison model)
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━ [2/2] Training MLP (deep learning comparison) ━━━")
scaler_mlp = StandardScaler()
# The MLP is kept as a comparison model for the capstone writeup, not as the
# default server model.
mlp_model  = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),   # Three layers give the comparison model enough capacity.
    activation="relu",
    solver="adam",
    alpha=1e-4,                           # Light regularization helps avoid memorizing the resampled data.
    batch_size=512,
    learning_rate_init=0.001,
    max_iter=50,                          # Enough epochs for a prototype without turning training into a wait.
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=False,
)
mlp_pipe = Pipeline([("scaler", scaler_mlp), ("model", mlp_model)])

t0 = time.time()
mlp_pipe.fit(X_train, y_train)
mlp_elapsed = time.time() - t0

y_pred_mlp    = mlp_pipe.predict(X_test)
mlp_metrics   = compute_metrics(y_test, y_pred_mlp, mlp_elapsed)
mlp_per_class = per_class_f1(y_test, y_pred_mlp, id_to_label)

print(f"   ✅ MLP done in {mlp_elapsed:.1f}s | Accuracy: {mlp_metrics['accuracy']}%")
print(classification_report(y_test, y_pred_mlp, zero_division=0))

# Save MLP separately so the server can optionally use it later.
joblib.dump({"pipeline": mlp_pipe, "id_to_label": id_to_label}, "vigil_mlp_pipeline.pkl")
print("   💾 Saved vigil_mlp_pipeline.pkl (MLP comparison model)")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Write model_stats.json  (read by the dashboard)
# ══════════════════════════════════════════════════════════════════════════════
# The dashboard reads this file directly, which keeps the web server from having
# to rerun training just to display model metrics.
stats = {
    "xgboost": {
        **xgb_metrics,
        "per_class_f1": xgb_per_class,
        "architecture": "XGBoost (n=250, depth=6, lr=0.1)",
        "type": "Gradient Boosted Trees",
    },
    "mlp": {
        **mlp_metrics,
        "per_class_f1": mlp_per_class,
        "architecture": "MLP (256 → 128 → 64, ReLU, Adam)",
        "type": "Deep Neural Network",
    },
    "dataset": {
        "total_samples":   int(len(df)),
        "train_samples":   int(X_train.shape[0]),
        "test_samples":    int(X_test.shape[0]),
        "features":        len(selected_features),
        "classes":         len(id_to_label),
    }
}

with open("model_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n✅ Saved model_stats.json")
print("\n━━━ FINAL COMPARISON ━━━")
print(f"  {'Metric':<12} {'XGBoost':>10} {'MLP':>10}")
print(f"  {'-'*34}")
for k in ["accuracy", "precision", "recall", "f1", "train_time_sec"]:
    unit = "s" if k == "train_time_sec" else "%"
    print(f"  {k:<12} {str(xgb_metrics[k])+unit:>10} {str(mlp_metrics[k])+unit:>10}")
print("\n🎉 All done! Run your server and visit /models to see the dashboard.")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# The confusion matrix image is generated as a submission artifact and as a
# quick visual check for which classes the production model confuses.
labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_xgb, display_labels=labels, xticks_rotation=45)
plt.title("Vigil.AI — XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")
