from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_explain_endpoint_ok():
    payload = {
        "age": 22,
        "income": 28000,
        "employment_length": 1,
        "dti": 0.55,
        "utilization": 0.90,
        "delinquencies": 2,
        "history_length": 1.5,
        "tx_30d_count": 8,
        "refund_rate_30d": 0.15,
        "active_days_30d": 5,
        "channel": "partner",
        "region": "SE",
        "product": "installment",
    }
    r = client.post("/v1/explain", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "model_version" in data
    assert "top_features" in data
    assert "reasons" in data
    assert len(data["reasons"]) >= 1
