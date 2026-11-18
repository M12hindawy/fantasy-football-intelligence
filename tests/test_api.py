"""
tests/test_api.py
=================
Integration tests for the FastAPI application.
"""
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "3.0"}

def test_predictions_endpoint():
    response = client.get("/predictions/FIX-TEST?home_team_id=H&away_team_id=A")
    if response.status_code != 200:
        print("Response JSON:", response.json())
    assert response.status_code == 200
    data = response.json()
    assert data["fixture_id"] == "FIX-TEST"
    assert "predictions" in data
    assert "p_home_win" in data["predictions"]

def test_fantasy_endpoint():
    response = client.get("/fantasy/recommendation?gameweek=1")
    assert response.status_code == 200
    data = response.json()
    assert "gameweek" in data
    assert "top_picks" in data
    assert isinstance(data["top_picks"], list)
