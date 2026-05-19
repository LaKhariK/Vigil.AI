import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_NAMES = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Bwd Pkt Len Max",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "ACK Flag Cnt",
]


@pytest.fixture(scope="module")
def model_module():
    if not (PROJECT_ROOT / "model.py").exists():
        pytest.skip("model.py is not present in this checkout")
    sys.path.insert(0, str(PROJECT_ROOT))
    return importlib.import_module("model")


@pytest.fixture()
def dummy_pipeline():
    class DummyScaler:
        mean_ = np.zeros(32)

    class DummyPipeline:
        named_steps = {"scaler": DummyScaler()}

    return DummyPipeline()


@pytest.fixture()
def sample_traffic_frame():
    return np.array(
        [
            [
                145032,
                12,
                7,
                6200,
                3180,
                1460,
                1460,
                516.7,
                454.3,
                64632.4,
                130.9,
                7640.6,
                1290.2,
                10980,
                12,
                8600.2,
                7300.4,
                0,
                0,
                0,
                0,
                384,
                224,
                82.7,
                48.2,
                492.1,
                231.4,
                53545.96,
                0,
                1,
                0,
                1,
            ],
            [
                9210,
                250,
                1,
                154000,
                40,
                1200,
                40,
                616.0,
                40.0,
                16728121.6,
                27252.9,
                36.8,
                14.1,
                102,
                1,
                37.0,
                0.0,
                1,
                0,
                0,
                0,
                8000,
                32,
                27144.4,
                108.6,
                613.7,
                144.0,
                20736.0,
                0,
                1,
                0,
                0,
            ],
        ],
        dtype=float,
    )


def _resolve_callable(module, *names):
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    raise AssertionError(f"None of these callables exist on model.py: {names}")


def _as_array(value):
    return np.asarray(value)


def test_sample_traffic_fixture_has_expected_feature_count(sample_traffic_frame):
    assert sample_traffic_frame.shape == (2, 32)
    assert len(FEATURE_NAMES) == 32


def test_preprocessing_handles_null_values(model_module, sample_traffic_frame, dummy_pipeline):
    preprocess = _resolve_callable(
        model_module, "preprocess_features", "preprocess_data", "preprocess_input", "preprocess"
    )
    sample_traffic_frame[0, FEATURE_NAMES.index("Flow Byts/s")] = np.nan

    processed = _as_array(preprocess(sample_traffic_frame, dummy_pipeline))

    assert not np.isnan(processed.astype(float)).any()


def test_preprocessing_preserves_batch_shape(model_module, sample_traffic_frame, dummy_pipeline):
    preprocess = _resolve_callable(
        model_module, "preprocess_features", "preprocess_data", "preprocess_input", "preprocess"
    )

    processed = _as_array(preprocess(sample_traffic_frame, dummy_pipeline))

    assert processed.shape[0] == 2
    assert processed.shape[1] == 32


def test_preprocessing_returns_numeric_matrix(model_module, sample_traffic_frame, dummy_pipeline):
    preprocess = _resolve_callable(
        model_module, "preprocess_features", "preprocess_data", "preprocess_input", "preprocess"
    )

    processed = _as_array(preprocess(sample_traffic_frame, dummy_pipeline))

    assert np.issubdtype(processed.dtype, np.number)


def test_model_loader_returns_predictable_model(model_module):
    load_model = _resolve_callable(model_module, "load_model", "load_pipeline", "get_model")

    try:
        loaded = load_model()
    except ModuleNotFoundError as exc:
        pytest.skip(f"Model dependency is not installed: {exc.name}")

    pipeline = loaded.get("pipeline") if isinstance(loaded, dict) else loaded

    assert callable(getattr(pipeline, "predict", None))


def test_prediction_returns_non_empty_result(model_module, sample_traffic_frame):
    predict = _resolve_callable(
        model_module, "predict_attack", "predict_traffic", "predict_class", "predict"
    )

    try:
        result = predict(sample_traffic_frame[0].tolist())
    except ModuleNotFoundError as exc:
        pytest.skip(f"Model dependency is not installed: {exc.name}")

    result_array = np.atleast_1d(result)

    assert result_array.size >= 1
    assert isinstance(result_array[0].item() if hasattr(result_array[0], "item") else result_array[0], (int, str))
