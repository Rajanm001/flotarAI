import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommendations_returns_top_10(client):
    payload = {
        "user_id": 900002,
        "name": "Pytest User",
        "gender": "Female",
        "dob": "1990-01-01",
        "interests": ["Music", "Art", "Books"],
        "city": "London",
        "country": "United Kingdom",
    }
    response = client.post("/recommendations", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["target_user_id"] == 900002
    assert body["candidate_pool_size"] == 100
    assert len(body["recommendations"]) == 10

    scores = [item["score"] for item in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_recommendations_rejects_empty_interests(client):
    payload = {
        "user_id": 900003,
        "name": "No Interests",
        "gender": "Male",
        "dob": "1990-01-01",
        "interests": [],
        "city": "London",
        "country": "United Kingdom",
    }
    response = client.post("/recommendations", json=payload)
    assert response.status_code == 422


def test_recommendations_rejects_unknown_interest(client):
    payload = {
        "user_id": 900004,
        "name": "Bad Interest",
        "gender": "Male",
        "dob": "1990-01-01",
        "interests": ["NotARealInterest"],
        "city": "London",
        "country": "United Kingdom",
    }
    response = client.post("/recommendations", json=payload)
    assert response.status_code == 422


def test_recommendations_rejects_blank_interest_string(client):
    payload = {
        "user_id": 900005,
        "name": "Blank Interest",
        "gender": "Male",
        "dob": "1990-01-01",
        "interests": [""],
        "city": "London",
        "country": "United Kingdom",
    }
    response = client.post("/recommendations", json=payload)
    assert response.status_code == 422


def test_recommendations_rejects_malformed_dob(client):
    payload = {
        "user_id": 900006,
        "name": "Bad Dob",
        "gender": "Male",
        "dob": "not-a-date",
        "interests": ["Music"],
        "city": "London",
        "country": "United Kingdom",
    }
    response = client.post("/recommendations", json=payload)
    assert response.status_code == 422
