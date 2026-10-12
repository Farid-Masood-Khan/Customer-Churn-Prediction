"""
preprocess.py
Data loading, cleaning, validation, and leak-free preprocessing pipeline.

CRITICAL DESIGN RULE (Coursera ML Specialization / Defensible ML):
Transformers (StandardScaler, OneHotEncoder) are fit ONLY on the training split (X_train).
X_test and any live inference samples are transformed using the fitted preprocessor.
No target leakage or test data leakage is permitted.
"""

from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Canonical feature definitions
NUMERIC_FEATURES: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load raw Telco Churn CSV from disk and validate basic schema.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Telco customer dataset:
    1. Drop customerID (non-predictive unique identifier).
    2. Parse TotalCharges from string to float; impute blank strings with 0.0 (tenure == 0).
    3. Standardize SeniorCitizen to string category or retain as categorical.
    4. Map Churn 'Yes'/'No' to binary 1/0 target.
    """
    df = df.copy()

    # Drop ID column if present
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # TotalCharges contains whitespace strings " " for 11 customers where tenure == 0
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # Logical imputation: customers with tenure 0 have accrued $0 in total charges
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Convert SeniorCitizen (0/1 int) to string for uniform categorical handling
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Encode target if present
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Construct an unfitted ColumnTransformer:
    - StandardScaler for continuous numerical features.
    - OneHotEncoder(handle_unknown='ignore', sparse_output=False) for categorical features.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract transformed feature names after preprocessor has been fit.
    """
    feature_names = []
    # Numeric features retain original names
    feature_names.extend(NUMERIC_FEATURES)
    # Categorical features get expanded one-hot column names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(list(cat_names))
    return feature_names


def prepare_splits(
    data_path: str | Path,
    test_size: float = 0.20,
    random_state: int = 42,
    save_preprocessor_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer, List[str]]:
    """
    Full end-to-end split and leak-free preprocessing:
    1. Clean data.
    2. Split into train & test with stratification on target.
    3. Fit preprocessor ONLY on X_train.
    4. Transform X_train and X_test.
    5. Optionally serialize fitted preprocessor to disk.

    Returns:
        (X_train_trans, X_test_trans, y_train, y_test, fitted_preprocessor, feature_names)
    """
    raw_df = load_raw_data(data_path)
    cleaned_df = clean_data(raw_df)

    X = cleaned_df.drop(columns=[TARGET_COLUMN])
    y = cleaned_df[TARGET_COLUMN]

    # Stratified split to preserve 26.5% positive churn ratio across splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()

    # CRITICAL: Fit ONLY on train split
    X_train_trans = preprocessor.fit_transform(X_train)
    # Transform test split using fitted train statistics
    X_test_trans = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    if save_preprocessor_path:
        out_path = Path(save_preprocessor_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, out_path)
        print(f"[preprocess] Saved fitted preprocessor to: {out_path}")

    return X_train_trans, X_test_trans, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    default_data = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    default_model_dir = Path(__file__).resolve().parent.parent / "models"
    preprocessor_file = default_model_dir / "preprocessor.joblib"

    print(f"[preprocess] Running verification on: {default_data}")
    X_tr, X_te, y_tr, y_te, prep, f_names = prepare_splits(
        default_data, save_preprocessor_path=preprocessor_file
    )
    print(f"[preprocess] Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
    print(f"[preprocess] Train churn rate: {y_tr.mean():.4f}, Test churn rate: {y_te.mean():.4f}")
    print(f"[preprocess] Transformed feature count: {len(f_names)}")
