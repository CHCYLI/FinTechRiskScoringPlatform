from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_score_endpoint_ok():
    payload = {
        "age": 35,
        "income": 65000,
        "employment_length": 5,
        "dti": 0.25,
        "utilization": 0.45,
        "delinquencies": 0,
        "history_length": 8,
        "tx_30d_count": 40,
        "refund_rate_30d": 0.02,
        "active_days_30d": 18,
        "channel": "organic",
        "region": "NE",
        "product": "card"
    }
    r = client.post("/v1/score", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["pd"] <= 1.0
    assert data["decision"] in ["Approve", "Review", "Reject"]
    assert "model_version" in data