import json
import sys
import traceback
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "vigil_pipeline.pkl"


@lru_cache(maxsize=1)
def load_model(model_path=MODEL_PATH):
    # Loading the joblib file is the slow part, so the server process reuses the
    # same model after the first prediction request.
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"] if isinstance(bundle, dict) and "pipeline" in bundle else bundle
    model = bundle.get("model", pipeline) if isinstance(bundle, dict) else pipeline
    return {"bundle": bundle, "pipeline": pipeline, "model": model}


def _expected_feature_count(pipeline, features):
    # The scaler remembers how many features it was trained on, which is safer
    # than hard-coding 32 in the prediction path.
    scaler = getattr(pipeline, "named_steps", {}).get("scaler")
    if scaler is not None and hasattr(scaler, "mean_"):
        return scaler.mean_.shape[0]
    return features.shape[1]


def preprocess_features(features, pipeline=None):
    # Browser inputs can arrive as a single flat list; sklearn expects rows.
    values = np.asarray(features, dtype=float)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if pipeline is None:
        pipeline = load_model()["pipeline"]

    expected = _expected_feature_count(pipeline, values)
    current = values.shape[1]

    # The web form and saved model can get out of sync during experiments. Pad
    # or trim rather than crashing the demo over missing extra fields.
    if current < expected:
        padding = np.zeros((values.shape[0], expected - current))
        return np.hstack([values, padding])
    if current > expected:
        return values[:, :expected]
    return values


def predict(features):
    # Keeping preprocessing beside prediction makes the API route forgiving of
    # raw frontend values.
    loaded = load_model()
    model = loaded["model"]
    processed = preprocess_features(features, loaded["pipeline"])
    predictions = model.predict(processed)
    return predictions


def predict_one(features):
    # The web UI needs both the numeric class and the student-friendly label.
    loaded = load_model()
    bundle = loaded["bundle"]
    pred = predict(features)[0]
    if hasattr(pred, "item"):
        pred = pred.item()

    id_to_label = bundle.get("id_to_label", {}) if isinstance(bundle, dict) else {}
    label = id_to_label.get(int(pred), pred)

    return {
        "prediction": int(pred),
        "label": str(label),
    }


def main():
    try:
        # Node sends one JSON payload through stdin, and this script returns one
        # JSON object through stdout so the boundary stays easy to debug.
        data = json.loads(sys.stdin.read())
        result = predict_one(data["features"])
        print(json.dumps(result))
    except Exception as exc:
        error = {"error": str(exc), "trace": traceback.format_exc()}
        print(json.dumps(error))


if __name__ == "__main__":
    main()
