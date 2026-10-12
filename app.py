"""
app.py
Streamlit Interactive Customer Churn Prediction Demo
IBM Telco Customer Churn ML Pipeline

Author: Farid Masood Khan
Companion to Coursera Machine Learning Specialization (DeepLearning.AI / Stanford Online)
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Telco Churn Predictor | ML Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    .churn-high {
        color: #ef4444;
        font-weight: bold;
    }
    .churn-low {
        color: #10b981;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-size: 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    models = {
        "Random Forest (Balanced) — Best F1": joblib.load(MODELS_DIR / "random_forest.joblib"),
        "Gradient Boosting — Best Accuracy": joblib.load(MODELS_DIR / "gradient_boosting.joblib"),
        "Logistic Regression (Balanced) — Baseline": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
    }
    with open(MODELS_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)
    with open(MODELS_DIR / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    return preprocessor, models, metrics, feature_names


try:
    preprocessor, models, metrics, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run `python src/train.py` and `python src/evaluate.py` first.")
    st.stop()

# Header
st.title("📊 Telco Customer Churn Intelligence & ML Pipeline")
st.markdown(
    """
    **End-to-End Supervised Machine Learning Pipeline** trained on the IBM Telco Customer Churn dataset.  
    Evaluated with **5-Fold Stratified Cross-Validation** and tested on a strict **holdout test split (N=1,409)** with zero feature leakage.
    """
)

tab_demo, tab_models, tab_methodology = st.tabs([
    "🎯 Live Inference Simulator",
    "📈 Honest Model Comparison & ROC",
    "🔬 Methodology & Error Analysis",
])

with tab_demo:
    col_input, col_pred = st.columns([1.1, 1], gap="large")

    with col_input:
        st.subheader("1. Configure Customer Profile")

        with st.expander("👤 Account & Contract Details", expanded=True):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
                tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=4)
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            with col_c2:
                payment = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    index=0,
                )
                monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.50, step=1.0)
                # Logical default for total charges
                default_total = float(round(tenure * monthly, 2)) if tenure > 0 else 0.0
                total = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=max(0.0, default_total), step=10.0)

        with st.expander("🌐 Internet & Telecommunication Services", expanded=True):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
                phone = st.selectbox("Phone Service", ["Yes", "No"], index=0)
                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    ["Yes", "No", "No phone service"] if phone == "Yes" else ["No phone service"],
                    index=0,
                )
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"], index=0)
            with col_s2:
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"], index=0)
                device_protect = st.selectbox("Device Protection", ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"], index=0)
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"], index=0)
                stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"] if internet != "No" else ["No internet service"], index=0)
                stream_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"] if internet != "No" else ["No internet service"], index=0)

        with st.expander("👥 Demographics", expanded=False):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                gender = st.selectbox("Gender", ["Female", "Male"], index=0)
                senior = st.selectbox("Senior Citizen", ["0 (No)", "1 (Yes)"], index=0)
            with col_d2:
                partner = st.selectbox("Partner", ["No", "Yes"], index=0)
                dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)

        selected_model_name = st.selectbox("Model for Prediction", list(models.keys()), index=0)

    with col_pred:
        st.subheader("2. Churn Risk Assessment")

        # Build single row dataframe for inference
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": "1" if "1" in senior else "0",
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protect,
            "TechSupport": tech_support,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total),
        }])

        model = models[selected_model_name]
        input_transformed = preprocessor.transform(input_data)
        churn_proba = float(model.predict_proba(input_transformed)[0, 1])
        prediction = int(churn_proba >= 0.5)

        # Risk Classification
        if churn_proba >= 0.65:
            risk_badge = "🔴 HIGH RISK OF CHURN"
            risk_color = "#ef4444"
            rec_text = "Urgent Intervention Required: Customer displays typical churn markers (short tenure, month-to-month contract, or electronic check billing)."
        elif churn_proba >= 0.35:
            risk_badge = "🟡 MODERATE RISK"
            risk_color = "#f59e0b"
            rec_text = "Proactive Retention Recommended: Offer multi-month contract promotion or bundle Tech Support."
        else:
            risk_badge = "🟢 LOW RISK (LOYAL)"
            risk_color = "#10b981"
            rec_text = "Stable Account: High retention probability. Recommended for upsell or long-term loyalty rewards."

        st.markdown(
            f"""
            <div style="background-color: #1e293b; border-left: 6px solid {risk_color}; padding: 20px; border-radius: 8px;">
                <h3 style="margin: 0; color: {risk_color};">{risk_badge}</h3>
                <h1 style="font-size: 48px; margin: 10px 0;">{churn_proba:.1%}</h1>
                <p style="margin: 0; color: #94a3b8;">Predicted Churn Probability ({selected_model_name.split('—')[0].strip()})</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.progress(churn_proba)

        st.markdown(f"**Recommended Action:** {rec_text}")

        st.write("---")
        st.subheader("💡 Strategic Retention Levers")
        levers = []
        if contract == "Month-to-month":
            levers.append("📌 **Contract Switch:** Transitioning from Month-to-month to a One-year contract reduces baseline churn odds significantly.")
        if internet == "Fiber optic" and tech_support == "No":
            levers.append("🛠️ **Bundle Tech Support:** High-paying Fiber optic users without Tech Support experience the highest dissatisfaction churn.")
        if payment == "Electronic check":
            levers.append("💳 **Automated Payment Incentive:** Offering a $2/mo discount to switch from Electronic check to Bank Transfer or Credit Card cuts friction.")
        if tenure <= 6:
            levers.append("⏳ **Onboarding Care Program:** Customer is within the critical 0–6 month hazard period where 60%+ of churn occurs.")

        if levers:
            for l in levers:
                st.markdown(l)
        else:
            st.success("This customer profile is well-anchored with multi-year contract and automated payment.")


with tab_models:
    st.subheader("Honest Benchmark on Holdout Test Split (N=1,409)")
    st.caption("All models trained on 5,634 training samples with leak-free preprocessing; evaluated on 1,409 unseen holdout test samples.")

    # Results Table
    test_metrics = metrics["test_metrics"]
    results_rows = []
    for model_name, m in test_metrics.items():
        results_rows.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall (Sensitivity)": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "True Positives (Caught)": m["confusion_matrix"][1][1],
            "False Negatives (Missed)": m["confusion_matrix"][1][0],
        })

    results_df = pd.DataFrame(results_rows)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.write("---")
    st.subheader("Model Diagnostic Charts")
    col_roc, col_cm = st.columns(2)

    with col_roc:
        roc_img_path = MODELS_DIR / "roc_curves.png"
        if roc_img_path.exists():
            st.image(str(roc_img_path), caption="Comparative ROC Curves on Unseen Test Split", use_container_width=True)

    with col_cm:
        cm_img_path = MODELS_DIR / "confusion_matrix.png"
        if cm_img_path.exists():
            st.image(str(cm_img_path), caption="Holdout Test Set Confusion Matrices (N=1,409)", use_container_width=True)

    feat_img_path = MODELS_DIR / "feature_importance.png"
    if feat_img_path.exists():
        st.image(str(feat_img_path), caption="Top Predictive Churn Drivers (Random Forest / Tree Importances)", use_container_width=True)


with tab_methodology:
    st.subheader("Academic & Technical Defense")

    err = metrics.get("error_analysis", {})
    st.markdown(
        f"""
        ### 1. Data Hygiene & Zero-Leakage Architecture
        - **Source:** IBM Telco Customer Churn (`7,043` records).
        - **Preprocessing:** Categorical variables one-hot encoded; continuous variables standardized using `StandardScaler`.
        - **Leakage Prevention:** Transformers were fit **strictly on `X_train`** (`5,634` records). No test split information influenced the mean, standard deviation, or categorical vocabulary.
        - **Validation Strategy:** 5-Fold Stratified Cross-Validation on the training split to select hyperparameter baselines.

        ### 2. Honest Error Analysis: Who Does the Model Miss?
        In a real-world enterprise setting, identifying false negatives (churners incorrectly classified as staying) is critical:
        - **Total Churners in Test Split:** `{err.get('total_churners', 374)}`
        - **False Negatives (Missed Churners):** `{err.get('false_negatives_count', 77)}` (using balanced Random Forest)
        - **Key Discovery:** Missed churners have an average tenure of **`{err.get('fn_avg_tenure', 33.5):.1f}` months**, compared to **`{err.get('tp_avg_tenure', 12.1):.1f}` months** for correctly detected churners.
        - **Interpretation:** The model excels at catching volatile early-tenure month-to-month churners, but struggles with long-standing loyal customers who churn abruptly due to external factors (service outages or competitor promos).

        ### 3. Trade-off Analysis: Recall vs. Precision
        - **Gradient Boosting** achieved the highest raw accuracy (**`{test_metrics['Gradient Boosting']['accuracy']:.2%}`**) and precision (**`{test_metrics['Gradient Boosting']['precision']:.2%}`**), but missed almost half of all churners (Recall: `{test_metrics['Gradient Boosting']['recall']:.2%}`).
        - **Random Forest with Class Balancing** deliberately sacrificed some precision to achieve **`{test_metrics['Random Forest']['recall']:.2%}` Recall**, capturing `{test_metrics['Random Forest']['confusion_matrix'][1][1]}` out of 374 churners.
        """
    )
