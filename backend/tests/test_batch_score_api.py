from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_batch_score_endpoint_ok():
    payload = [
        {
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
            "product": "card",
        },
        {
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
        },
    ]
    r = client.post("/v1/score/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert 0 <= data["results"][0]["pd"] <= 1
