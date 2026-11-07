"""
test_preprocessor.py
Unit tests for Telco Churn data cleaning, imputation, and leak-free transformations.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import pytest

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocess import (
    clean_data,
    build_preprocessor,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    ID_COLUMN,
)


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "customerID": ["7590-VHVEG", "5575-GNVDE", "3668-QPYBK"],
        "gender": ["Female", "Male", "Male"],
        "SeniorCitizen": [0, 0, 0],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "No"],
        "tenure": [1, 34, 2],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "No"],
        "InternetService": ["DSL", "DSL", "DSL"],
        "OnlineSecurity": ["No", "Yes", "Yes"],
        "OnlineBackup": ["Yes", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "No", "No"],
        "StreamingTV": ["No", "No", "No"],
        "StreamingMovies": ["No", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": ["29.85", "1889.5", "108.15"],
        "Churn": ["No", "No", "Yes"],
    })


def test_clean_data_drops_customer_id(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    assert ID_COLUMN not in cleaned.columns


def test_clean_data_handles_blank_total_charges():
    df = pd.DataFrame({
        "customerID": ["TEST-1"],
        "tenure": [0],
        "MonthlyCharges": [20.0],
        "TotalCharges": [" "],
        "Churn": ["No"]
    })
    cleaned = clean_data(df)
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert not cleaned["TotalCharges"].isna().any()


def test_clean_data_maps_target_to_binary(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    assert TARGET_COLUMN in cleaned.columns
    assert set(cleaned[TARGET_COLUMN].unique()).issubset({0, 1})
    assert cleaned[TARGET_COLUMN].tolist() == [0, 0, 1]


def test_preprocessor_leak_free_transformation(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    X = cleaned.drop(columns=[TARGET_COLUMN])

    # Fit preprocessor on X
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Check output shape
    assert X_transformed.shape[0] == 3
    assert X_transformed.shape[1] > len(NUMERIC_FEATURES)  # OHE expands categoricals
    assert not np.isnan(X_transformed).any()


def test_preprocessor_handles_unseen_categories(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    X_train = cleaned.drop(columns=[TARGET_COLUMN])

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Unseen test sample with novel category
    X_unseen = X_train.copy().iloc[[0]]
    X_unseen["PaymentMethod"] = "Crypto Token"  # Never seen in train

    # Should not raise ValueError due to handle_unknown='ignore'
    X_transformed = preprocessor.transform(X_unseen)
    assert X_transformed.shape[0] == 1
    assert not np.isnan(X_transformed).any()
