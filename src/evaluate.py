"""
evaluate.py
Honest evaluation on unseen test split, metric comparison, visualization generation,
and in-depth error analysis.

Acceptance Criteria Enforced:
1. Metrics evaluated strictly on X_test (transformed using train preprocessor).
2. Honest reporting: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
3. Confusion matrices, ROC curves, and feature attribution saved as high-res figures.
4. Error analysis identifying exactly which customer segment false negatives represent.
"""

from pathlib import Path
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
import joblib

from preprocess import prepare_splits, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Compute standard binary classification performance metrics on holdout test set.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def plot_roc_curves(
    eval_results: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    output_path: Path,
):
    """
    Plot comparative ROC Curves for all candidate models.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    colors = {"Logistic Regression": "#2563eb", "Random Forest": "#059669", "Gradient Boosting": "#d97706"}

    for name, res in eval_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        auc_score = res["roc_auc"]
        color = colors.get(name, "#4b5563")
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", linewidth=2.2, color=color)

    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)", alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) — Unseen Test Set (N=1,409)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[evaluate] Saved ROC Curves to: {output_path}")


def plot_confusion_matrices(
    eval_results: Dict[str, Dict[str, Any]],
    output_path: Path,
):
    """
    Plot side-by-side Confusion Matrices for all candidate models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    model_names = list(eval_results.keys())
    cmaps = ["Blues", "Greens", "Oranges"]

    for i, name in enumerate(model_names):
        cm = np.array(eval_results[name]["confusion_matrix"])
        ax = axes[i]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmaps[i % len(cmaps)],
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"{name}\nAcc: {eval_results[name]['accuracy']:.3f} | Rec: {eval_results[name]['recall']:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Class (0=Stay, 1=Churn)", fontsize=10)
        ax.set_ylabel("Actual Class (0=Stay, 1=Churn)", fontsize=10)
        ax.set_xticklabels(["Stay (0)", "Churn (1)"])
        ax.set_yticklabels(["Stay (0)", "Churn (1)"])

    plt.suptitle("Confusion Matrix Comparison on Holdout Test Split (N=1,409)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved Confusion Matrices to: {output_path}")


def plot_feature_importance(
    model: Any,
    feature_names: list,
    output_path: Path,
    top_n: int = 15,
):
    """
    Plot top feature importances (or logistic regression absolute coefficients).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        metric_label = "Gini Feature Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        metric_label = "Absolute Logistic Coefficient |β|"
    else:
        return

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(top_features))

    ax.barh(y_pos, top_vals, align="center", color="#3b82f6", edgecolor="#1d4ed8")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(metric_label, fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Predictive Features for Customer Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[evaluate] Saved Feature Importance to: {output_path}")


def conduct_error_analysis(
    data_path: Path,
    best_model: Any,
    preprocessor: Any,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Error analysis: inspect False Negatives (actual churners missed by model)
    and False Positives (loyal customers incorrectly flagged).
    """
    from preprocess import load_raw_data, clean_data, TARGET_COLUMN
    from sklearn.model_selection import train_test_split

    raw_df = load_raw_data(data_path)
    clean_df = clean_data(raw_df)

    X = clean_df.drop(columns=[TARGET_COLUMN])
    y = clean_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_test_trans = preprocessor.transform(X_test)
    preds = best_model.predict(X_test_trans)
    probas = best_model.predict_proba(X_test_trans)[:, 1] if hasattr(best_model, "predict_proba") else preds

    analysis_df = X_test.copy()
    analysis_df["Actual_Churn"] = y_test.values
    analysis_df["Predicted_Churn"] = preds
    analysis_df["Churn_Probability"] = probas

    # Segment errors
    false_negatives = analysis_df[(analysis_df["Actual_Churn"] == 1) & (analysis_df["Predicted_Churn"] == 0)]
    false_positives = analysis_df[(analysis_df["Actual_Churn"] == 0) & (analysis_df["Predicted_Churn"] == 1)]
    true_positives = analysis_df[(analysis_df["Actual_Churn"] == 1) & (analysis_df["Predicted_Churn"] == 1)]
    true_negatives = analysis_df[(analysis_df["Actual_Churn"] == 0) & (analysis_df["Predicted_Churn"] == 0)]

    summary = {
        "total_test_samples": len(analysis_df),
        "total_churners": int((analysis_df["Actual_Churn"] == 1).sum()),
        "true_positives_count": len(true_positives),
        "false_negatives_count": len(false_negatives),
        "false_positives_count": len(false_positives),
        "true_negatives_count": len(true_negatives),
        "fn_avg_tenure": float(false_negatives["tenure"].mean()),
        "fn_avg_monthly_charges": float(false_negatives["MonthlyCharges"].mean()),
        "fn_contract_distribution": false_negatives["Contract"].value_counts(normalize=True).to_dict(),
        "tp_avg_tenure": float(true_positives["tenure"].mean()),
        "tp_avg_monthly_charges": float(true_positives["MonthlyCharges"].mean()),
        "tp_contract_distribution": true_positives["Contract"].value_counts(normalize=True).to_dict(),
    }

    return summary


def run_evaluation_pipeline(
    data_path: str | Path,
    models_dir: str | Path,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Main evaluation pipeline.
    """
    data_path = Path(data_path)
    models_dir = Path(models_dir)

    # Load fitted preprocessor
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    with open(models_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    # Retrieve test split strictly using the same seed
    X_train_trans, X_test_trans, y_train, y_test, _, _ = prepare_splits(
        data_path=data_path,
        random_state=random_state,
    )

    model_files = {
        "Logistic Regression": models_dir / "logistic_regression.joblib",
        "Random Forest": models_dir / "random_forest.joblib",
        "Gradient Boosting": models_dir / "gradient_boosting.joblib",
    }

    eval_results = {}
    print("\n[evaluate] Evaluating models on unseen test split (N=1,409)...")
    for name, path in model_files.items():
        if path.exists():
            mod = joblib.load(path)
            res = evaluate_model(mod, X_test_trans, y_test)
            eval_results[name] = res
            print(
                f"  -> {name:20s} | "
                f"Accuracy: {res['accuracy']:.4f} | "
                f"Precision: {res['precision']:.4f} | "
                f"Recall: {res['recall']:.4f} | "
                f"F1: {res['f1']:.4f} | "
                f"ROC-AUC: {res['roc_auc']:.4f}"
            )

    # Visualizations
    roc_plot_file = models_dir / "roc_curves.png"
    cm_plot_file = models_dir / "confusion_matrix.png"
    feat_plot_file = models_dir / "feature_importance.png"

    plot_roc_curves(eval_results, y_test, roc_plot_file)
    plot_confusion_matrices(eval_results, cm_plot_file)

    # Feature importance of Gradient Boosting / Best Model
    best_model = joblib.load(models_dir / "best_model.joblib")
    plot_feature_importance(best_model, feature_names, feat_plot_file)

    # Error analysis
    error_summary = conduct_error_analysis(data_path, best_model, preprocessor, random_state=random_state)
    print("\n[evaluate] Error Analysis Summary (Best Model):")
    print(f"  - Total test churners: {error_summary['total_churners']}")
    print(f"  - False Negatives (churners missed): {error_summary['false_negatives_count']}")
    print(f"  - False Positives (loyal users flagged): {error_summary['false_positives_count']}")
    print(f"  - False Negative Avg Tenure: {error_summary['fn_avg_tenure']:.1f} months (vs {error_summary['tp_avg_tenure']:.1f} for True Positives)")
    print(f"  - False Negative Avg Monthly Charges: ${error_summary['fn_avg_monthly_charges']:.2f}")

    # Clean serializable dictionary without numpy arrays
    metrics_to_save = {}
    for name, res in eval_results.items():
        metrics_to_save[name] = {
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "roc_auc": res["roc_auc"],
            "confusion_matrix": res["confusion_matrix"],
        }

    full_evaluation_payload = {
        "test_metrics": metrics_to_save,
        "error_analysis": error_summary,
    }

    metrics_json_file = models_dir / "metrics.json"
    with open(metrics_json_file, "w") as f:
        json.dump(full_evaluation_payload, f, indent=2)
    print(f"[evaluate] Saved metrics and error analysis to: {metrics_json_file}")

    return full_evaluation_payload


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    out_dir = base_dir / "models"

    print("==================================================")
    print("  TELCO CUSTOMER CHURN — EVALUATION PIPELINE     ")
    print("==================================================")
    run_evaluation_pipeline(data_file, out_dir)
