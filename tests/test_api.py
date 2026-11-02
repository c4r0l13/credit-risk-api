import copy

import pandas as pd

from app.drift import psi, psi_status
from app.features import build_features
from app.model import ModelService
from tests.conftest import synthetic_raw


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "test-0001"}


def test_score_returns_valid_decision(client, application):
    r = client.post("/v1/score", json=application)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] in {"approve", "decline"}
    assert "x-request-id" in r.headers


def test_declines_include_reasons(client, application):
    risky = copy.deepcopy(application) | {"pay_status": [3, 3, 3, 3, 3, 3]}
    body = client.post("/v1/score", json=risky).json()
    if body["decision"] == "decline":
        assert body["reasons"], "every decline must come with adverse-action reasons"


def test_risk_is_monotonic_in_delinquency(client, application):
    pds = []
    for d in range(0, 4):
        app = copy.deepcopy(application) | {"pay_status": [d, 0, 0, 0, 0, 0]}
        pds.append(client.post("/v1/score", json=app).json()["probability_of_default"])
    assert pds == sorted(pds)


def test_validation_rejects_bad_payloads(client, application):
    assert client.post("/v1/score", json=application | {"LIMIT_BAL": -5}).status_code == 422
    assert client.post("/v1/score", json=application | {"pay_status": [0, 0]}).status_code == 422
    assert client.post("/v1/score", json=application | {"unexpected": 1}).status_code == 422


def test_batch_and_drift_endpoint(client, application):
    apps = [application | {"application_id": f"a{i}"} for i in range(250)]
    r = client.post("/v1/score/batch", json={"applications": apps})
    assert r.status_code == 200 and len(r.json()["decisions"]) == 250
    drift = client.get("/v1/monitoring/drift").json()
    assert drift["window_size"] >= 250
    # every request was the same applicant -> the window is far from the reference
    assert any(s == "significant" for s in drift["status"].values())


def test_metrics_endpoint(client, application):
    client.post("/v1/score", json=application)
    text = client.get("/metrics").text
    assert "credit_decisions_total" in text


def test_model_service_rejects_feature_mismatch(artifact_dir, tmp_path):
    import json
    import shutil

    for f in ("model.txt", "reference.csv"):
        shutil.copy(artifact_dir / f, tmp_path / f)
    meta = json.loads((artifact_dir / "metadata.json").read_text())
    meta["features"] = ["something_else"]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    try:
        ModelService(tmp_path)
    except RuntimeError:
        return
    raise AssertionError("feature mismatch must fail fast at startup")


def test_psi_detects_shift_including_discrete_features():
    raw, _ = synthetic_raw(2000, seed=1)
    ref = build_features(raw)
    same = build_features(synthetic_raw(2000, seed=2)[0])
    shifted = raw.copy()
    shifted["PAY_0"] = shifted["PAY_0"] + 1
    cur = build_features(shifted)
    assert psi(ref["utilization_last"], same["utilization_last"]) < 0.1
    assert psi_status(psi(ref["PAY_0"], cur["PAY_0"])) == "significant"


def test_build_features_requires_raw_columns():
    try:
        build_features(pd.DataFrame({"LIMIT_BAL": [1]}))
    except ValueError as e:
        assert "Missing raw columns" in str(e)
    else:
        raise AssertionError
