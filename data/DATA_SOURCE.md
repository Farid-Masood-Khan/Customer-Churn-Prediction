# Dataset Documentation & Source Provenance

## Dataset Name
**IBM Telco Customer Churn** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)

## Source & Attribution
- **Original Publisher:** IBM Community / Kaggle Public Datasets
- **Direct Source:** Public distribution of the standard IBM sample analytics dataset
- **Context:** Tabular records of 7,043 telecommunications customers in California in Q3.
- **Licensing/Terms:** Public IBM sample dataset released for educational and predictive modeling benchmarks.
- **Statement of Provenance:** This dataset is a public benchmark provided by IBM. It was **not** collected, scraped, or curated privately by Farid Masood Khan. All rights remain with the original creators.

## Target Variable
- `Churn`: Whether the customer left within the last month (`Yes` or `No`).
- Binary classification target (~26.5% positive churn rate, indicating moderate class imbalance).

## Feature Summary (21 columns total)
- **Identifier:**
  - `customerID`: Unique alphanumeric string identifier (dropped during modeling to avoid leakage/spurious memorization).
- **Demographics:**
  - `gender`: Female / Male
  - `SeniorCitizen`: Binary flag (0 = No, 1 = Yes)
  - `Partner`: Whether customer has a partner (Yes / No)
  - `Dependents`: Whether customer has dependents (Yes / No)
- **Account & Contract Info:**
  - `tenure`: Number of months the customer has stayed with the company (Continuous numeric, 0–72)
  - `Contract`: Contract term (`Month-to-month`, `One year`, `Two year`)
  - `PaperlessBilling`: Paperless billing option (`Yes`, `No`)
  - `PaymentMethod`: `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`
  - `MonthlyCharges`: Current monthly amount charged (Continuous numeric, $18.25–$118.75)
  - `TotalCharges`: Cumulative charges over tenure (Numeric; contains 11 blank spaces for new customers with tenure = 0, converted to 0.0 during preprocessing)
- **Services Subscribed:**
  - `PhoneService`: `Yes` / `No`
  - `MultipleLines`: `No phone service`, `No`, `Yes`
  - `InternetService`: `DSL`, `Fiber optic`, `No`
  - `OnlineSecurity`: `Yes`, `No`, `No internet service`
  - `OnlineBackup`: `Yes`, `No`, `No internet service`
  - `DeviceProtection`: `Yes`, `No`, `No internet service`
  - `TechSupport`: `Yes`, `No`, `No internet service`
  - `StreamingTV`: `Yes`, `No`, `No internet service`
  - `StreamingMovies`: `Yes`, `No`, `No internet service`

## Data Hygiene & Integrity Rules
1. Never fit encoders, scalers, or imputation logic on test or validation splits.
2. The 11 records with whitespace in `TotalCharges` represent customers with `tenure = 0` whose total bill is logically `0.0`.
3. Metrics are evaluated on an unseen test split (20% holdout, stratified by churn class with a fixed random state).
