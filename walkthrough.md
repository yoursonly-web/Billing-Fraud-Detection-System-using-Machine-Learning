# Walkthrough — Billing Fraud Detection System

## What Was Done

1. **Cleaned and Preprocessed Data**:
   - Dropped 50 rows with missing targets (`is_fraud` was null).
   - Checked for and removed duplicate rows.
   - Fixed 394 logically inconsistent readings by swapping `previous_reading` and `meter_reading` where the previous reading was larger.
   - Created binary `_missing` flags for all numeric columns to preserve missingness as a signal.
   - Imputed numeric columns using their **median** value. Justified this as median is robust to outliers compared to the mean.
   - Confirmed outlier ranges using the IQR method (outliers kept as potential fraud indicators).

2. **Exploratory Data Analysis (EDA)**:
   - Evaluated group differences via t-test to check if averages differed significantly between fraud and non-fraud groups. `billing_amount`, `avg_last_6_months`, `payment_delay_days`, and `location_risk_score` show highly significant differences.
   - Analyzed correlations with `is_fraud` and explicitly documented the answers to the key questions: "What patterns indicate fraud?" and "Which features are most important?".

3. **Feature Engineering**:
   - Engineered 7 new features (using correct specification names where requested):
     - `billing_ratio` = `billing_amount / avg_last_6_months` (capped at 50)
     - `billing_deviation` = `billing_amount - avg_last_6_months` (replaces `billing_diff`)
     - `usage_diff` = `meter_reading - previous_reading` (replaces `meter_diff`)
     - `delay_risk_interaction` = `payment_delay_days * location_risk_score`
     - `billing_per_unit` = `billing_amount / usage_diff`
     - `high_billing_flag` = 1 if `billing_ratio > 2` else 0
     - `payment_delay_category` (binned categorical representation)
   - Explained why these features are important.

4. **Model Training & Comparison**:
   - Split dataset using a **80/20 stratified train/test split**.
   - Trained four models: **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**.
   - Balanced class weights for all models to handle label ratio.
   - Selected the **Random Forest** as the best model due to its high recall, precision, and F1-score.

5. **Threshold Tuning Analysis**:
   - Calculated precision, recall, and F1-score for thresholds ranging from 0.1 to 0.95.
   - Explained why Recall is critical in fraud detection.

6. **Streamlit App Deployment (Step 10)**:
   - Created `app.py`, a premium dark-themed dashboard.
   - Updated all feature references to match the new schema (`usage_diff`, `billing_deviation`).
   - Page views: Dashboard, EDA & Analysis, Model Performance, Predict Fraud, Threshold Tuning, and Monitoring.

7. **Model Monitoring & Info (Step 11)**:
   - Added a dedicated **📈 Monitoring** page in `app.py` displaying model metadata, training configuration parameters, dataset summary stats, complete feature documentation (21 features explained), interactive Pareto chart for feature importance deep dive, and baseline stats.

8. **Insights & Business Understanding (Task 7)**:
   - Added detailed answers addressing what causes fraud, high-risk customer profiles, and recommendations for reducing fraud.

---

## Validation & Results

### Model Comparison Metrics (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 89.47% | 95.28% | 89.63% | 92.37% | 95.56% |
| Decision Tree | 95.26% | 96.32% | 97.04% | 96.68% | 94.88% |
| **Random Forest [BEST]** | **97.37%** | **99.24%** | **97.04%** | **98.13%** | **98.65%** |
| XGBoost | 96.84% | 98.50% | 97.04% | 97.76% | 98.67% |

### Verification Details
- **Training execution**: Completed successfully. All models, scalers, and metadata saved in `e:\Data Analytics\models\`.
- **Streamlit App**: Local server successfully launched. Updated to use the correct `billing_deviation` and `usage_diff` feature keys.
- **Jupyter Notebook**: Fully generated and executed at `e:\Data Analytics\analysis_model.ipynb` containing all requested additions (Dataset Summary, Imputation justification, Outlier boxplots, Key EDA answers, Feature descriptions, Recall description, and Step 10 Insights).
