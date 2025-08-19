# test_pipeline.py
import pytest
import joblib
import pandas as pd
import os

# Path to your pipeline
MODEL_PATH = "app/models/catboost_smote_pipeline.joblib"


@pytest.fixture(scope="module")
def pipeline():
    """Load the trained pipeline once for all tests."""
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


def test_pipeline_prediction_shape(pipeline):
    """Check if pipeline returns a single prediction for one input."""
    sample_input = pd.DataFrame([{
        "Power": 100.0,
        "OSF": 1.0,
        "PWF": 1.0,
        "HDF": 1.0,
        "TWF": 1.0,
        "Torque [Nm]": 50.0,
        "Rotational speed [rpm]": 1500.0,
        "Temp_Difference": 20.0,
    }])
    
    prediction = pipeline.predict(sample_input)
    assert prediction.shape == (1,)


def test_pipeline_prediction_values(pipeline):
    """Check if pipeline output is only 0 or 1 (classification)."""
    sample_input = pd.DataFrame([{
        "Power": 250.0,
        "OSF": 2.0,
        "PWF": 2.0,
        "HDF": 2.0,
        "TWF": 2.0,
        "Torque [Nm]": 70.0,
        "Rotational speed [rpm]": 2000.0,
        "Temp_Difference": 25.0,
    }])
    
    prediction = pipeline.predict(sample_input)[0]
    assert prediction in [0, 1]


def test_pipeline_multiple_inputs(pipeline):
    """Test prediction on multiple rows of input."""
    sample_input = pd.DataFrame([
        {
            "Power": 120.0,
            "OSF": 1.2,
            "PWF": 1.0,
            "HDF": 1.1,
            "TWF": 1.0,
            "Torque [Nm]": 60.0,
            "Rotational speed [rpm]": 1600.0,
            "Temp_Difference": 18.0,
        },
        {
            "Power": 300.0,
            "OSF": 2.5,
            "PWF": 2.2,
            "HDF": 2.0,
            "TWF": 2.3,
            "Torque [Nm]": 90.0,
            "Rotational speed [rpm]": 2100.0,
            "Temp_Difference": 30.0,
        },
    ])
    
    predictions = pipeline.predict(sample_input)
    assert len(predictions) == 2
    assert all(p in [0, 1] for p in predictions)
