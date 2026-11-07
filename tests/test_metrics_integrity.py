"""
test_metrics_integrity.py
Verifies mathematical and statistical integrity of reported benchmark metrics.
"""

from pathlib import Path
import json
import pytest


@pytest.fixture
def metrics_data():
    path = Path(__file__).resolve().parent.parent / "models" / "metrics.json"
    with open(path, "r") as f:
        return json.load(f)


def test_holdout_test_metrics_within_expected_bounds(metrics_data):
    test_metrics = metrics_data["test_metrics"]
    
    # Random Forest check
    rf = test_metrics["Random Forest"]
    assert 0.74 <= rf["accuracy"] <= 0.76
    assert 0.78 <= rf["recall"] <= 0.81
    assert 0.83 <= rf["roc_auc"] <= 0.86

    # Gradient Boosting check
    gb = test_metrics["Gradient Boosting"]
    assert 0.79 <= gb["accuracy"] <= 0.82
    assert 0.83 <= gb["roc_auc"] <= 0.86


def test_error_analysis_counts_match_confusion_matrix(metrics_data):
    rf_cm = metrics_data["test_metrics"]["Random Forest"]["confusion_matrix"]
    tn, fp = rf_cm[0]
    fn, tp = rf_cm[1]

    err = metrics_data["error_analysis"]
    assert err["true_positives_count"] == tp
    assert err["false_negatives_count"] == fn
    assert err["false_positives_count"] == fp
    assert err["true_negatives_count"] == tn
    assert err["total_test_samples"] == tn + fp + fn + tp
