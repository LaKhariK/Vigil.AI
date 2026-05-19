import joblib, numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
# Use a non-interactive backend so the script can run from a terminal or test
# task without opening a desktop plotting window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the already-trained production pipeline so this script only rebuilds the
# evaluation split and redraws the confusion matrix.
bundle = joblib.load("vigil_pipeline.pkl")
pipe = bundle["pipeline"]
id_to_label = bundle["id_to_label"]

# Pull the same parquet dataset used during training.
base_path = os.path.join(os.path.dirname(__file__), "cicddos2019")
dfs = []
for root, _, files in os.walk(base_path):
    for f in files:
        if f.lower().endswith(".parquet"):
            dfs.append(pd.read_parquet(os.path.join(root, f), engine="pyarrow"))

df = pd.concat(dfs, ignore_index=True)

# Match the training label cleanup so the matrix reflects the same seven
# dashboard categories.
label_series = df["Label"].astype(str)
replacements = {
    "DrDoS":"DDoS","WebDDoS":"DDoS","UDP-lag":"DDoS","UDPLag":"DDoS",
    "LDAP":"DDoS","MSSQL":"DDoS","NetBIOS":"DDoS","NTP":"DDoS","TFTP":"DDoS",
    "Syn":"DoS","UDP":"DoS","Portmap":"PortScan"
}
for old, new in replacements.items():
    label_series = label_series.str.replace(old, new, regex=False)

id_map = {"Benign":0,"DDoS":1,"DoS":2,"PortScan":3,"Botnet":4,"Infiltration":5,"WebAttack":6}
y = label_series.map(lambda x: next((k for k in id_map if k.lower() in x.lower()), None))
y = y.fillna("Benign").map(id_map).astype(int)

# Keep feature order identical to the saved model; a shuffled column order would
# make the predictions meaningless even if the values are valid.
selected_features = [
    "Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts",
    "Fwd Pkt Len Max","Bwd Pkt Len Max","Fwd Pkt Len Mean","Bwd Pkt Len Mean","Flow Byts/s",
    "Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Mean",
    "Bwd IAT Mean","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags",
    "Fwd Header Len","Bwd Header Len","Fwd Pkts/s","Bwd Pkts/s","Pkt Len Mean","Pkt Len Std",
    "Pkt Len Var","FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","ACK Flag Cnt"
]
X = df.select_dtypes(include=[np.number]).reindex(columns=selected_features, fill_value=0)

# Rebuild the balanced test split used for evaluation so the confusion matrix is
# easier to read across minority attack classes.
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
_, X_test, _, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

y_pred = pipe.predict(X_test)

# Only show classes that actually appear in the test/prediction output. This
# avoids matplotlib label-count errors when a class is absent in a quick run.
present = sorted(np.unique(np.concatenate([y_test, y_pred])))
labels = [id_to_label[i] for i in present]

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels, ax=ax, colorbar=True)
plt.title("Vigil.AI — XGBoost Confusion Matrix", fontsize=13, pad=12)
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Done — confusion_matrix.png saved!")
