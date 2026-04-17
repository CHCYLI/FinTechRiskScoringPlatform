from app.services.inference import score_batch, score_one

def test_score_one_returns_pd_and_decision():
    payload = {
        "age": 30,
        "income": 50000,
        "employment_length": 3,
        "dti": 0.3,
        "utilization": 0.6,
        "delinquencies": 1,
        "history_length": 6,
        "tx_30d_count": 25,
        "refund_rate_30d": 0.03,
        "active_days_30d": 12,
        "channel": "organic",
        "region": "NE",
        "product": "card"
    }
    out = score_one(payload)
    assert 0.0 <= out["pd"] <= 1.0
    assert out["decision"] in ["Approve", "Review", "Reject"]


def test_score_batch_returns_expected_shape():
    payload = [
        {
            "age": 30,
            "income": 50000,
            "employment_length": 3,
            "dti": 0.3,
            "utilization": 0.6,
            "delinquencies": 1,
            "history_length": 6,
            "tx_30d_count": 25,
            "refund_rate_30d": 0.03,
            "active_days_30d": 12,
            "channel": "organic",
            "region": "NE",
            "product": "card",
        }
    ]
    out = score_batch(payload)
    assert "results" in out
    assert len(out["results"]) == 1
    assert 0 <= out["results"][0]["pd"] <= 1
