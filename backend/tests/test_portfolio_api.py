from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_portfolio_summary_by_region():
    r = client.get("/v1/portfolio/summary", params={"group_by": "region"})
    assert r.status_code == 200
    data = r.json()

    assert "model_version" in data
    assert data["group_by"] == "region"
    assert "rows" in data

    if data["rows"]:
        first = data["rows"][0]
        assert "group" in first
        assert "count" in first
        assert "avg_pd" in first
        assert "approve_count" in first
        assert "review_count" in first
        assert "reject_count" in first
        assert "approve_rate" in first
        assert "review_rate" in first
        assert "reject_rate" in first


def test_portfolio_summary_with_filter():
    r = client.get(
        "/v1/portfolio/summary",
        params={"group_by": "product", "region": "NE"},
    )
    assert r.status_code == 200
    data = r.json()

    assert data["group_by"] == "product"
    assert data["filters"]["region"] == "NE"


def test_portfolio_summary_invalid_limit():
    r = client.get("/v1/portfolio/summary", params={"group_by": "region", "limit": 0})
    assert r.status_code == 422
