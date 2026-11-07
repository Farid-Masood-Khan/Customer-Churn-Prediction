"""
test_models.py
Unit tests for saved model artifacts, inference pipelines, and probability predictions.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def models_dir():
    return Path(__file__).resolve().parent.parent / "models"


def test_saved_artifacts_exist(models_dir):
    required_files = [
        "best_model.joblib",
        "preprocessor.joblib",
        "logistic_regression.joblib",
        "random_forest.joblib",
        "gradient_boosting.joblib",
        "feature_names.json",
        "metrics.json",
        "cv_results.json",
    ]
    for filename in required_files:
        assert (models_dir / filename).exists(), f"Missing artifact: {filename}"


def test_model_inference_pipeline(models_dir):
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    model = joblib.load(models_dir / "best_model.joblib")

    # Synthetic inference payload matching Telco schema
    sample = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": ["0"],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [85.5],
        "TotalCharges": [1026.0],
    })

    X_transformed = preprocessor.transform(sample)
    proba = model.predict_proba(X_transformed)[0]

    assert len(proba) == 2
    assert 0.0 <= proba[0] <= 1.0
    assert 0.0 <= proba[1] <= 1.0
    assert np.isclose(proba[0] + proba[1], 1.0)


def test_feature_names_match_preprocessor_output(models_dir):
    with open(models_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    out_features = preprocessor.get_feature_names_out()

    assert len(feature_names) == len(out_features)
