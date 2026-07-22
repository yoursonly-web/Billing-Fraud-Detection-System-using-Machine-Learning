# Billing Fraud Detection System — Project Report

This report outlines the design, implementation, evaluation, and business insights of the Billing Fraud Detection system following the requested 11-step analytical pipeline.

---

## 1. Executive Summary

We built an end-to-end Machine Learning pipeline that processes utility/billing data, cleans anomalies, engineers features, compares 4 ML algorithms, and deploys a web application for utility companies.
The final **Random Forest** model achieves **97.37% Accuracy**, **99.24% Precision**, and **97.04% Recall** on unseen test data.

---

## 2. Pipeline Implementation Details

### TASK 1: Data Understanding
- **Objective**: Flag billing anomalies, detect abnormal consumption, and minimize manual audit workloads.
- **Dataset Summary**: The source dataset `billing_fraud_dataset (1).csv` contains 1,000 raw customer billing transaction records with 9 columns.
- **Features**:
  - `customer_id` (Identifier): Unique customer account number.
  - `billing_amount` (Numeric): Current period charge (₹).
  - `avg_last_6_months` (Numeric): Customer's historical baseline bill (₹).
  - `payment_delay_days` (Numeric): Days payment was delayed.
  - `meter_reading` (Numeric): Current consumption meter value.
  - `previous_reading` (Numeric): Previous period meter value.
  - `location_risk_score` (Numeric): Geographic risk score (0.0 to 1.0).
  - `num_complaints` (Numeric): Complaints filed by the customer.
  - `is_fraud` (Target): 1.0 for fraudulent, 0.0 for legitimate.
- **Data Quality Gaps**: Missing values exist in ~5% of every column, target labels were missing for 50 records, and 394 records contained logically inconsistent meter readings where `previous_reading > meter_reading`.

### TASK 2: Data Cleaning
- **Target Missingness**: Dropped 50 rows where the target label `is_fraud` was null, resulting in 950 clean records.
- **Duplicate Removal**: Checked for and removed fully duplicate rows to ensure model training was not biased by duplicate entries.
- **Logical Consistency Reading Swap**: Swapped the values of `previous_reading` and `meter_reading` for 394 records where `previous_reading > meter_reading`, correcting data entry errors.
- **Missing Value Imputation**: Imputed numeric columns using their **median** value (computed on training data). Median was selected over the mean because it is robust against the extreme billing amount outliers present in our dataset.
- **Missingness Flags**: Created binary indicator flags (`*_missing`) for all numeric columns prior to imputation to preserve the fact that a value was missing as a predictive signal.
- **Outlier Treatment**: Confirmed outlier ranges using the Interquartile Range (IQR) method. Outliers in billing amounts were flagged but retained in the training set as they represent potential fraud signals.

### TASK 3: Exploratory Data Analysis (EDA)
- **Class Balance**: 71.3% fraud rate (677 cases) vs. 28.7% legitimate (273 cases) in the clean dataset.
- **Statistical Significance Testing (T-tests)**:
  - `billing_amount` (p < 0.0001, Highly Significant)
  - `avg_last_6_months` (p < 0.0001, Highly Significant)
  - `payment_delay_days` (p < 0.0001, Highly Significant)
  - `location_risk_score` (p < 0.0001, Highly Significant)
  - `meter_reading` (p = 0.5481, Not Significant)
  - `num_complaints` (p = 0.3471, Not Significant)
- **Key Patterns Indicating Fraud**:
  - Abnormally high billing amount relative to low historical averages.
  - Long payment delays.
  - Customers situated in high-risk zones.

### TASK 4: Feature Engineering
We engineered 7 new features to capture relative deviations, usage metrics, and risk interaction indicators:
1. **`billing_ratio`**: `billing_amount / avg_last_6_months` (capped at 50 to prevent extreme ratios). Strongest correlation with fraud ($r = 0.589$ via its flag).
2. **`billing_deviation`**: Raw financial difference (`billing_amount - avg_last_6_months`).
3. **`usage_diff`**: Consumption delta (`meter_reading - previous_reading`). Swapping readings for consistency ensures this is a non-negative usage metric.
4. **`delay_risk_interaction`**: Interaction risk score (`payment_delay_days * location_risk_score`).
5. **`billing_per_unit`**: Cost per unit of consumption (`billing_amount / usage_diff`).
6. **`high_billing_flag`**: Binary indicator (1 if `billing_ratio > 2` else 0).
7. **`payment_delay_category`**: Delays binned into 4 categorical risk buckets.

### TASK 5 & 6: Model Building and Evaluation
We split the clean dataset using an 80/20 stratified split and trained 4 distinct classification models. The performance metrics evaluated on the test set are summarized below:

| Metric | Logistic Regression | Decision Tree | Random Forest | XGBoost |
|---|---|---|---|---|
| **Accuracy** | 89.47% | 95.26% | **97.37%** | 96.84% |
| **Precision** | 95.28% | 96.32% | **99.24%** | 98.50% |
| **Recall (Fraud Caught)** | 89.63% | **97.04%** | **97.04%** | **97.04%** |
| **F1 Score** | 92.37% | 96.68% | **98.13%** | 97.76% |
| **ROC-AUC** | 95.56% | 94.88% | **98.65%** | **98.67%** |

*The **Random Forest** model was selected as the final production model due to superior Accuracy (97.37%), Precision (99.24%), and F1-Score (98.13%).*

#### Why is Recall Crucial in Fraud Detection?
Recall is the primary metric optimized in fraud detection because of the **asymmetry of costs**:
- **False Negatives (Missed Fraud)**: Direct revenue leak for the utility company since fraudulent consumption goes unpaid.
- **False Positives (False Alarms)**: Low cost, requiring only a brief manual record review by an auditor.
Optimizing for high recall ensures that we capture 97.0% of actual fraud cases, prioritizing revenue protection.

---

## 3. TASK 7: Insights & Business Understanding

### What Causes Fraud?
- **Unexplained Billing Spikes**: Fraud is heavily characterized by sudden spikes in billing amounts that are completely out of line with historical baseline averages (`billing_deviation` and `billing_ratio` are top features by Gini importance).
- **Geographic Risk**: Fraud cases cluster in specific high-risk zones, suggesting organized local tampering or lack of monitoring in certain areas.
- **Payment Delay Correlation**: Fraudulent customers consistently pay late, possibly to delay detection or due to deliberate refusal to pay.

### Which Customers Are High Risk?
Auditors should prioritize accounts meeting these thresholds:
1. **`billing_ratio` > 2.0x**: Current bill is double the historical 6-month average.
2. **`billing_deviation` > ₹3,000**: Raw increase of more than ₹3,000 above baseline.
3. **`payment_delay_days` > 20 days**: Persistent payment delays.
4. **`location_risk_score` > 0.7**: Accounts registered in designated high-risk zip codes.
5. **Inconsistent Readings**: Accounts with historic consumption reading resets (`previous_reading > meter_reading`).

### How Can the Company Reduce Fraud?
1. **Automated ML Auditing**: Integrate the Random Forest classifier into the billing system to automatically place high-probability fraud flags on hold for review, catching 97% of leaks.
2. **Data Validation Rules**: Implement real-time database constraints on meter readings to instantly block or flag any inputs where `previous_reading > meter_reading` at the time of entry.
3. **Dynamic Threshold Auditing**: Adjust the Streamlit decision threshold based on auditor capacity (e.g., lower threshold to 0.30 to capture more fraud if audit resources increase).
4. **Targeted Field Inspections**: Send inspectors to conduct physical audits of billing audits located in zones where the location risk score is > 0.7.

---

## 4. Run Instructions

### 1. Model Training
To retrain the models and regenerate pipeline files:
```bash
.venv\Scripts\python.exe train_model.py
```

### 2. Streamlit Web App
To run the interactive dark-themed dashboard:
```bash
.venv\Scripts\streamlit.exe run app.py
```
*App will run locally at [http://localhost:8501](http://localhost:8501).*
