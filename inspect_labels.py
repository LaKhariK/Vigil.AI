import os
import pandas as pd

# Quick dataset audit helper: it checks each parquet file for the label-like
# column names that the training script needs.
base_path = "./cicddos2019"
for f in os.listdir(base_path):
    if f.endswith(".parquet"):
        full_path = os.path.join(base_path, f)
        print(f"\n📄 File: {f}")
        try:
            df = pd.read_parquet(full_path, engine="pyarrow")
            # Dataset exports are not always consistent, so this searches for a
            # few likely target column names instead of assuming "Label".
            label_col = [c for c in df.columns if "label" in c.lower() or "attack" in c.lower() or "category" in c.lower()]
            if label_col:
                col = label_col[0]
                print("   → Label column:", col)
                print("   → Unique values:", df[col].unique()[:10])
            else:
                print("   ⚠️ No label-like column found!")
        except Exception as e:
            print("   ❌ Could not read file:", e)
