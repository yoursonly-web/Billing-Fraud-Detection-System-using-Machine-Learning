# Billing Fraud Detection System — ML Pipeline + Streamlit App

## Overview

Build an end-to-end billing fraud detection system following the user's 11-step pipeline. The system will train multiple ML models on the provided `billing_fraud_dataset (1).csv` (1,001 rows, 9 columns) and deploy a premium Streamlit web app for real-time fraud prediction.

In addition, we will create a comprehensive Jupyter notebook (`analysis_model.ipynb`) detailing every step of the analysis, including data cleaning, EDA, statistical testing, feature engineering, model training/evaluation for 4 algorithms, threshold tuning, and production baseline logging. The notebook will contain all markdown explanations, code blocks, and fully rendered outputs (plots, charts, tables, and text).

---

## User Review Required

Document anything that requires user review or feedback, for example, breaking changes or significant design decisions. Use GitHub alerts (IMPORTANT/WARNING/CAUTION) to highlight critical items.

> [!IMPORTANT]
> - **Execution of the Notebook**: To embed all plots, tables, and print statements directly inside the `.ipynb` file as requested, we will run the notebook programmatically inside the virtual environment using `nbformat` and `nbclient`/`ipykernel` and save it.
> - **Replacing the Existing Notebook**: We will overwrite the existing incomplete `analysis_model.ipynb` with this new, comprehensive version.

---

## Open Questions

> [!NOTE]
> Please review the following design questions. They will not block the initial implementation but are provided for clarity:
> 1. **Plot styles**: We will use a clean, professional Seaborn style (`whitegrid`) for the notebook visualizations, which aligns with standard analytical practices. Let us know if you prefer a dark-themed notebook layout.
> 2. **Library Installation**: To programmatically execute and render the notebook, we will install `nbformat`, `nbclient`, and `ipykernel` in the virtual environment. Let us know if you have any constraints on installing these packages.

---

## Proposed Changes

### [Component Name] Jupyter Notebook Deliverable

#### [MODIFY] [analysis_model.ipynb](file:///e:/Data%20Analytics/analysis_model.ipynb)
Create a fully detailed, executed notebook including:
1. **Title & System Introduction**: Outline of the 11-step billing fraud detection pipeline.
2. **Setup**: Import packages, set random seed (42), and configure style.
3. **Data Loading**: Load raw dataset, verify dimensions (1001x9), and inspect initial rows.
4. **Data Cleaning**:
   - Duplicate customer ID analysis.
   - Missingness analysis (percent missing per column).
   - Creation of binary `_missing` flags to capture missingness signals.
   - Median imputation for numeric features.
5. **Exploratory Data Analysis (EDA)**:
   - Target variable class balance pie chart.
   - Distribution plots (histograms + box plots) for billing features segmented by fraud status.
   - Statistical significance testing (Independent T-tests) comparing fraud vs. non-fraud means.
   - Correlation analysis (pearson correlation heatmap).
6. **Feature Engineering**:
   - Implement `billing_ratio`, `billing_diff`, `meter_diff`, `delay_risk_interaction`, `billing_per_unit`, `high_billing_flag`, and `payment_delay_category`.
   - Re-evaluate correlation coefficients with target class.
7. **Pipeline Splitting & Scaling**:
   - Stratified train/test split (80/20).
   - Scaling validation using `StandardScaler`.
8. **Progressive Model Training**:
   - Logistic Regression (scaled inputs).
   - Decision Tree Classifier.
   - Random Forest Classifier (balanced weights).
   - XGBoost Classifier (weighted scale).
9. **Comprehensive Evaluation**:
   - Side-by-side Confusion Matrices.
   - Classification Reports (Precision, Recall, F1, Accuracy, Support).
   - Combined ROC Curve.
   - Combined Precision-Recall (PR) Curve.
   - Feature Importance Bar Chart for the final selected model.
10. **Threshold Tuning Analysis**:
    - Computing metrics across thresholds 0.1 to 0.95.
    - Threshold trade-off curve (Precision, Recall, F1 vs. Threshold).

#### [NEW] [generate_notebook.py](file:///e:/Data%20Analytics/generate_notebook.py)
A temporary Python script in the artifacts/scratch folder to programmatically construct, execute, and write the `.ipynb` file.

---

## Verification Plan

### Automated Tests
- Programmatically execute the Python script to build the notebook.
- Verify the generated `analysis_model.ipynb` contains the output cells (images, tables, and text).
- Check that the notebook compiles and executes without syntax or runtime errors.

### Manual Verification
- Open the notebook in a Jupyter environment or check the structure to ensure all cells are properly formatted.
