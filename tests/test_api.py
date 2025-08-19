# tests/test_api.py
import sys
import os
import pytest
from fastapi.testclient import TestClient
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# ✅ Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # now this will work safely

client = TestClient(app)


def test_homepage():
    """Test if the homepage (/) loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_no_failure():
    """Test prediction endpoint with valid input (should return HTML with prediction)."""
    response = client.post(
        "/predict",
        data={
            "Power": 100.0,
            "OSF": 1.0,
            "PWF": 1.0,
            "HDF": 1.0,
            "TWF": 1.0,
            "Torque_Nm": 50.0,
            "Rotational_speed_rpm": 1500.0,
            "Temp_difference": 20.0,
        },
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Machine Failure" in response.text or "No Failure" in response.text


def test_predict_invalid_input():
    """Test prediction endpoint with invalid input (should fail with 422)."""
    response = client.post(
        "/predict",
        data={
            "Power": "invalid",  # wrong type
            "OSF": 1.0,
            "PWF": 1.0,
            "HDF": 1.0,
            "TWF": 1.0,
            "Torque_Nm": 50.0,
            "Rotational_speed_rpm": 1500.0,
            "Temp_difference": 20.0,
        },
    )
    assert response.status_code == 422  # validation error
