"""
train.py
Model training, 5-fold cross-validation, hyperparameter selection, and serialization.

Models compared:
1. Logistic Regression (L2 regularized, class_weight='balanced')
2. Random Forest Classifier (ensemble of decision trees, class_weight='balanced')
3. Gradient Boosting Classifier (sequential boosted trees)

All models are validated using 5-Fold Stratified Cross-Validation on the training split,
followed by full fitting on X_train and persistence to disk.
"""

from pathlib import Path
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib

from preprocess import prepare_splits


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Define candidate models matching Coursera Course 1-2 supervised learning curriculum:
    - Linear baseline with log-odds sigmoid.
    - Non-linear bagging ensemble (Random Forest).
    - Sequential boosting ensemble (Gradient Boosting).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=random_state,
        ),
    }


def perform_cross_validation(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Run 5-Fold Stratified Cross-Validation on training data.
    Evaluates: ROC-AUC, F1-Score, Accuracy, Precision, Recall.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
    }

    cv_results = {}
    print(f"\n[train] Running {cv_folds}-Fold Stratified Cross-Validation on training set (N={len(y_train)})...")

    for name, model in models.items():
        print(f"  -> Cross-validating: {name}...")
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_results[name] = {
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "cv_f1_mean": float(np.mean(scores["test_f1"])),
            "cv_f1_std": float(np.std(scores["test_f1"])),
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
            "cv_precision_mean": float(np.mean(scores["test_precision"])),
            "cv_precision_std": float(np.std(scores["test_precision"])),
            "cv_recall_mean": float(np.mean(scores["test_recall"])),
            "cv_recall_std": float(np.std(scores["test_recall"])),
        }
        print(
            f"     ROC-AUC: {cv_results[name]['cv_roc_auc_mean']:.4f} (+/- {cv_results[name]['cv_roc_auc_std']:.4f}) | "
            f"F1: {cv_results[name]['cv_f1_mean']:.4f} | "
            f"Recall: {cv_results[name]['cv_recall_mean']:.4f}"
        )

    return cv_results


def fit_and_save_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: pd.Series,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Fit each candidate model on the entire training set (X_train) and save artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted_models = {}

    filename_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
    }

    for name, model in models.items():
        print(f"[train] Fitting {name} on full X_train...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        model_file = output_dir / filename_map[name]
        joblib.dump(model, model_file)
        print(f"  -> Saved {name} to: {model_file}")

    return fitted_models


def run_training_pipeline(
    data_path: str | Path,
    models_dir: str | Path,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full training pipeline driver.
    """
    data_path = Path(data_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    preprocessor_path = models_dir / "preprocessor.joblib"
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_splits(
        data_path=data_path,
        random_state=random_state,
        save_preprocessor_path=preprocessor_path,
    )

    # Save feature names list for inference and interpretation
    with open(models_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    candidate_models = get_candidate_models(random_state=random_state)
    cv_summary = perform_cross_validation(
        models=candidate_models,
        X_train=X_train,
        y_train=y_train,
        cv_folds=5,
        random_state=random_state,
    )

    fitted_models = fit_and_save_models(
        models=candidate_models,
        X_train=X_train,
        y_train=y_train,
        output_dir=models_dir,
    )

    # Select best model based on CV ROC-AUC
    best_name = max(cv_summary.keys(), key=lambda k: cv_summary[k]["cv_roc_auc_mean"])
    best_model = fitted_models[best_name]
    best_model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"\n[train] Best model by CV ROC-AUC: '{best_name}' -> Saved to {best_model_path}")

    # Persist CV summary
    cv_file = models_dir / "cv_results.json"
    with open(cv_file, "w") as f:
        json.dump(cv_summary, f, indent=2)

    return {
        "cv_summary": cv_summary,
        "best_model_name": best_name,
        "fitted_models": fitted_models,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    out_dir = base_dir / "models"

    print("==================================================")
    print("  TELCO CUSTOMER CHURN — MODEL TRAINING PIPELINE ")
    print("==================================================")
    result = run_training_pipeline(data_file, out_dir)
    print("\n[train] Training stage complete!")
