
Billing Fraud Detection System using Machine Learning

Machine learning system to detect fraudulent activity in utility billing data — from raw data through a trained, deployable classifier and an interactive web dashboard.

📋 Overview

This project builds an end-to-end fraud detection pipeline for utility billing records. It covers data cleaning and exploratory analysis, feature engineering, training and comparing multiple classification models, and deploying the best-performing model through a Streamlit web application for real-time predictions.

✨ Features
Full exploratory data analysis (EDA) on billing transaction data
Feature engineering tailored to fraud-indicative billing patterns
Comparison of multiple classifiers to select the best-performing model
Threshold tuning for optimized precision/recall trade-off
Trained Random Forest model selected for production use
Interactive Streamlit dashboard for live fraud predictions
🛠️ Tech Stack
Language: Python
Notebook Environment: Jupyter Notebook
ML Libraries: scikit-learn, pandas, numpy
Web App: Streamlit
Visualization: matplotlib / seaborn (update if different)
📁 Project Structure
Billing-Fraud-Detection-System-using-Machine-Learning/
│
├── analysis_model.ipynb        # EDA, feature engineering, model training & comparison
├── train_model.py              # Script to train and save the production model
├── app.py                      # Streamlit web app for live predictions
├── models/                     # Saved/serialized trained model(s)
├── billing_fraud_dataset.csv   # Sample billing dataset used for training
├── billing_fraud_report.md     # Detailed project report
├── implementation_plan.md      # Project planning and implementation notes
├── walkthrough.md              # Step-by-step walkthrough of the pipeline
├── requirements.txt            # Python dependencies
└── README.md
📊 Dataset

The dataset used (billing_fraud_dataset.csv) contains sample utility billing records with features engineered to capture patterns associated with fraudulent billing behavior. See billing_fraud_report.md for a full breakdown of the fields and preprocessing steps.

🧠 Methodology
Data Cleaning — handled missing values, outliers, and inconsistent entries in the raw billing data.
Exploratory Data Analysis (EDA) — analyzed distributions, correlations, and class imbalance between fraudulent and legitimate records.
Feature Engineering — derived new features from billing patterns to improve model signal.
Model Training & Comparison — trained and evaluated four classification algorithms on the processed dataset.
Threshold Tuning — adjusted the decision threshold to balance precision and recall for the fraud-detection use case.
Model Selection — selected Random Forest as the final production model based on evaluation metrics.

Full details and code are in analysis_model.ipynb.

📈 Results
Metric	Score
Accuracy	[add value]
Precision	[add value]
Recall	[add value]
F1-Score	[add value]
ROC-AUC	[add value]

See billing_fraud_report.md for confusion matrices, ROC curves, and detailed evaluation across all four models compared.

🚀 Getting Started
Prerequisites
Python 3.8+
pip

Installation

git clone https://github.com/yoursonly-web/Billing-Fraud-Detection-System-using-Machine-Learning.git
cd Billing-Fraud-Detection-System-using-Machine-Learning
pip install -r requirements.txt


Training the Model

python train_model.py
Running the Web App

streamlit run app.py

Then open the local URL shown in your terminal (typically http://localhost:8501) to use the dashboard.

📓 Notebook

For the full exploratory analysis, feature engineering, and model comparison process, open:

bash
jupyter notebook analysis_model.ipynb
📄 Additional Documentation
billing_fraud_report.md — Detailed project report
implementation_plan.md — Planning and implementation notes
walkthrough.md — Step-by-step pipeline walkthrough
