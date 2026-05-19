import http.client
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_FEATURES = [
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
]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(port, method, path, body=None, headers=None):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=payload, headers=request_headers)
    response = conn.getresponse()
    raw = response.read().decode("utf-8")
    conn.close()

    try:
        data = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        data = raw

    return response.status, dict(response.getheaders()), data


@pytest.fixture(scope="module")
def api_server():
    port = _free_port()
    env = {**os.environ, "PORT": str(port)}
    venv_scripts = PROJECT_ROOT / "venv" / "Scripts"
    env["PATH"] = f"{venv_scripts}{os.pathsep}{env.get('PATH', '')}"
    process = subprocess.Popen(
        ["node", "server.js"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.time() + 10
    while time.time() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            pytest.skip(f"server.js did not start: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                break
        except OSError:
            time.sleep(0.1)
    else:
        process.terminate()
        pytest.skip("server.js did not accept connections within 10 seconds")

    yield port

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture()
def auth_cookie(api_server):
    username = f"pytest_{uuid.uuid4().hex}"
    status, headers, data = _request(
        api_server,
        "POST",
        "/auth/register",
        {"username": username, "password": "pytest-password-123"},
    )

    assert status == 200, data
    assert data == {"ok": True}
    return headers["Set-Cookie"].split(";", 1)[0]


def test_root_redirects_to_login_page(api_server):
    status, headers, _ = _request(api_server, "GET", "/")

    assert status == 302
    assert headers["Location"] == "/login.html"


def test_predict_requires_authentication(api_server):
    status, headers, _ = _request(api_server, "POST", "/predict", {"features": SAMPLE_FEATURES})

    assert status == 302
    assert headers["Location"] == "/login.html"


def test_predict_rejects_missing_features(api_server, auth_cookie):
    status, _, data = _request(
        api_server,
        "POST",
        "/predict",
        {},
        headers={"Cookie": auth_cookie},
    )

    assert status == 400
    assert data == {"error": "No features provided"}


def test_predict_accepts_32_feature_payload(api_server, auth_cookie):
    status, _, data = _request(
        api_server,
        "POST",
        "/predict",
        {"features": SAMPLE_FEATURES},
        headers={"Cookie": auth_cookie},
    )

    assert status == 200
    assert isinstance(data, dict)
    assert "prediction" in data or "error" in data


def test_predict_accepts_short_feature_payload(api_server, auth_cookie):
    status, _, data = _request(
        api_server,
        "POST",
        "/predict",
        {"features": SAMPLE_FEATURES[:6]},
        headers={"Cookie": auth_cookie},
    )

    assert status == 200
    assert isinstance(data, dict)
    assert "prediction" in data or "error" in data


def test_predict_response_is_non_empty_json(api_server, auth_cookie):
    status, _, data = _request(
        api_server,
        "POST",
        "/predict",
        {"features": SAMPLE_FEATURES},
        headers={"Cookie": auth_cookie},
    )

    assert status == 200
    assert data
    assert isinstance(data, dict)
