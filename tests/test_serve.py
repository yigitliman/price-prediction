import pytest
from fastapi.testclient import TestClient

from src.serve import app

SAMPLE_HOUSE = {
    "MedInc": 5.0,
    "HouseAge": 20.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.1,
    "Population": 1200.0,
    "AveOccup": 3.0,
    "Latitude": 37.5,
    "Longitude": -122.0,
}

EXPENSIVE_HOUSE = {
    "MedInc": 15.0,
    "HouseAge": 5.0,
    "AveRooms": 10.0,
    "AveBedrms": 1.0,
    "Population": 800.0,
    "AveOccup": 2.5,
    "Latitude": 37.8,
    "Longitude": -122.4,
}

CHEAP_HOUSE = {
    "MedInc": 1.5,
    "HouseAge": 40.0,
    "AveRooms": 3.0,
    "AveBedrms": 1.5,
    "Population": 2000.0,
    "AveOccup": 4.0,
    "Latitude": 36.0,
    "Longitude": -119.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_price_with_interval(client):
    response = client.post("/predict", json=SAMPLE_HOUSE)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert "price_low" in data
    assert "price_high" in data
    assert data["predicted_price"] > 0
    assert data["price_low"] <= data["predicted_price"] <= data["price_high"]


def test_expensive_house_costs_more(client):
    r1 = client.post("/predict", json=EXPENSIVE_HOUSE).json()["predicted_price"]
    r2 = client.post("/predict", json=CHEAP_HOUSE).json()["predicted_price"]
    assert r1 > r2


def test_batch_predict(client):
    response = client.post("/predict/batch", json=[SAMPLE_HOUSE, EXPENSIVE_HOUSE, CHEAP_HOUSE])
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert "avg_predicted_price" in data
    assert len(data["results"]) == 3


def test_batch_predict_empty_returns_400(client):
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 400


def test_logs_returns_list(client):
    response = client.get("/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_stats_structure(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "avg_price" in data
