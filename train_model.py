"""
Billing Fraud Detection System - ML Training Pipeline
=====================================================
Steps 3–9: Data Cleaning, EDA, Feature Engineering, 
Train/Test Split, Model Training, Evaluation, Threshold Tuning

Author: Automated ML Pipeline
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score,
    recall_score, accuracy_score
)
from scipy import stats
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not installed. Skipping XGBoost model.")

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "billing_fraud_dataset (1).csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

print("=" * 70)
print("  BILLING FRAUD DETECTION - ML TRAINING PIPELINE")
print("=" * 70)
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Data: {DATA_PATH}")
print("=" * 70)

# ============================================================
# STEP 2: DATA COLLECTION
# ============================================================
print("\n[STEP 2] Loading dataset...")
df_raw = pd.read_csv(DATA_PATH)
print(f"  -> Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(f"  -> Columns: {list(df_raw.columns)}")

# ============================================================
# STEP 3: DATA CLEANING
# ============================================================
print("\n[STEP 3] Data Cleaning...")

df = df_raw.copy()

# 3a. Missing values analysis
missing_report = {}
for col in df.columns:
    n_missing = df[col].isna().sum()
    pct = n_missing / len(df) * 100
    missing_report[col] = {'count': int(n_missing), 'pct': round(pct, 2)}
    if n_missing > 0:
        print(f"  -> {col}: {n_missing} missing ({pct:.1f}%)")

# 3b. Drop rows where target (is_fraud) is missing
n_before = len(df)
df = df.dropna(subset=['is_fraud'])
n_dropped_target = n_before - len(df)
print(f"\n  -> Dropped {n_dropped_target} rows with missing target (is_fraud)")
print(f"  -> Remaining: {len(df)} rows")

# 3c. Check and remove duplicate rows
n_full_duplicates = df.duplicated().sum()
print(f"  -> Duplicate rows found: {n_full_duplicates}")
if n_full_duplicates > 0:
    df = df.drop_duplicates()
    print(f"  -> Removed duplicate rows. Remaining: {len(df)} rows")

n_missing_ids = df['customer_id'].isna().sum()
n_duplicate_ids = df['customer_id'].dropna().duplicated().sum()
print(f"  -> Missing customer_ids: {n_missing_ids}")
print(f"  -> Duplicate customer_ids: {n_duplicate_ids}")

# 3c-2. Data Consistency: Swap previous_reading and meter_reading if previous_reading > meter_reading
inconsistent_mask = df['previous_reading'] > df['meter_reading']
n_inconsistent = inconsistent_mask.sum()
if n_inconsistent > 0:
    df.loc[inconsistent_mask, ['meter_reading', 'previous_reading']] = \
        df.loc[inconsistent_mask, ['previous_reading', 'meter_reading']].values
    print(f"  -> Fixed {n_inconsistent} logical inconsistency rows by swapping readings")

# 3d. Create missing flags BEFORE imputation (missingness as a signal)
feature_cols_for_flags = ['billing_amount', 'avg_last_6_months', 'payment_delay_days',
                          'meter_reading', 'previous_reading', 'location_risk_score', 'num_complaints']
for col in feature_cols_for_flags:
    df[f'{col}_missing'] = df[col].isna().astype(int)
    n_flag = df[f'{col}_missing'].sum()
    if n_flag > 0:
        print(f"  -> Created {col}_missing flag ({n_flag} flagged)")

# 3e. Impute missing numeric values with median
numeric_cols = ['billing_amount', 'avg_last_6_months', 'payment_delay_days',
                'meter_reading', 'previous_reading', 'location_risk_score', 'num_complaints']
medians = {}
for col in numeric_cols:
    median_val = df[col].median()
    medians[col] = float(median_val)
    n_filled = df[col].isna().sum()
    df[col] = df[col].fillna(median_val)
    if n_filled > 0:
        print(f"  -> Filled {col} with median = {median_val:.2f} ({n_filled} values)")

# 3f. Outlier detection (IQR) - flag but don't remove (fraud IS the outlier)
outlier_info = {}
for col in ['billing_amount', 'avg_last_6_months', 'payment_delay_days']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_info[col] = {'Q1': float(Q1), 'Q3': float(Q3), 'IQR': float(IQR),
                          'lower': float(lower), 'upper': float(upper), 'n_outliers': int(n_outliers)}
    print(f"  -> {col}: {n_outliers} outliers (IQR method) - kept as potential fraud signals")

print(f"\n  [SUCCESS] Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 4: EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n[STEP 4] Exploratory Data Analysis...")

# 4a. Class balance
fraud_counts = df['is_fraud'].value_counts()
fraud_rate = df['is_fraud'].mean() * 100
print(f"  -> Class distribution: Fraud={int(fraud_counts.get(1, 0))}, "
      f"Non-Fraud={int(fraud_counts.get(0, 0))}")
print(f"  -> Fraud rate: {fraud_rate:.1f}%")

# 4b. Descriptive statistics
desc_stats = df[numeric_cols].describe().to_dict()

# 4c. Fraud vs Non-Fraud comparison
eda_comparison = {}
for col in numeric_cols:
    fraud_vals = df[df['is_fraud'] == 1][col]
    non_fraud_vals = df[df['is_fraud'] == 0][col]
    
    # T-test for significance
    t_stat, p_value = stats.ttest_ind(fraud_vals, non_fraud_vals, equal_var=False)
    significant = "YES" if p_value < 0.05 else "NO"
    
    eda_comparison[col] = {
        'fraud_mean': float(fraud_vals.mean()),
        'non_fraud_mean': float(non_fraud_vals.mean()),
        'fraud_median': float(fraud_vals.median()),
        'non_fraud_median': float(non_fraud_vals.median()),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': significant
    }
    print(f"  -> {col}: Fraud mean={fraud_vals.mean():.1f}, Non-Fraud mean={non_fraud_vals.mean():.1f}, "
          f"p={p_value:.4f} ({significant})")

# 4d. Correlation with target
correlations = {}
for col in numeric_cols:
    corr = df[col].corr(df['is_fraud'])
    correlations[col] = float(corr)
    print(f"  -> Correlation {col} <-> is_fraud: {corr:.3f}")

# 4e. Save EDA data
eda_data = {
    'missing_report': missing_report,
    'outlier_info': outlier_info,
    'fraud_counts': {str(k): int(v) for k, v in fraud_counts.items()},
    'fraud_rate': float(fraud_rate),
    'desc_stats': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in desc_stats.items()},
    'eda_comparison': eda_comparison,
    'correlations': correlations,
    'class_distribution': {'fraud': int(fraud_counts.get(1, 0)), 'non_fraud': int(fraud_counts.get(0, 0))},
    'total_rows_raw': int(df_raw.shape[0]),
    'total_rows_clean': int(df.shape[0]),
    'n_dropped_target': int(n_dropped_target),
}

# ============================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================
print("\n[STEP 5] Feature Engineering...")

# 5a. billing_ratio - is this bill unusual?
df['billing_ratio'] = df['billing_amount'] / df['avg_last_6_months'].replace(0, np.nan)
df['billing_ratio'] = df['billing_ratio'].fillna(df['billing_ratio'].median())
# Cap extreme ratios to prevent inf issues
df['billing_ratio'] = df['billing_ratio'].clip(0, 50)
print(f"  -> Created billing_ratio (mean={df['billing_ratio'].mean():.2f})")

# 5b. billing_deviation - raw difference
df['billing_deviation'] = df['billing_amount'] - df['avg_last_6_months']
print(f"  -> Created billing_deviation (mean={df['billing_deviation'].mean():.2f})")

# 5c. usage_diff - usage delta
df['usage_diff'] = df['meter_reading'] - df['previous_reading']
print(f"  -> Created usage_diff (mean={df['usage_diff'].mean():.2f})")

# 5d. delay_risk_interaction - combined risk signal
df['delay_risk_interaction'] = df['payment_delay_days'] * df['location_risk_score']
print(f"  -> Created delay_risk_interaction (mean={df['delay_risk_interaction'].mean():.2f})")

# 5e. billing_per_unit - cost efficiency check
df['billing_per_unit'] = df['billing_amount'] / df['usage_diff'].replace(0, np.nan)
df['billing_per_unit'] = df['billing_per_unit'].fillna(df['billing_per_unit'].median())
df['billing_per_unit'] = df['billing_per_unit'].clip(-100, 100)  # Cap extreme values
print(f"  -> Created billing_per_unit (mean={df['billing_per_unit'].mean():.2f})")

# 5f. Additional derived features
df['high_billing_flag'] = (df['billing_ratio'] > 2).astype(int)
print(f"  -> Created high_billing_flag ({df['high_billing_flag'].sum()} flagged)")

df['payment_delay_category'] = pd.cut(df['payment_delay_days'], 
                                       bins=[-1, 7, 14, 21, 30],
                                       labels=[0, 1, 2, 3]).astype(float).fillna(2)
print(f"  -> Created payment_delay_category")

# Correlations of new features
print("\n  New feature correlations with is_fraud:")
new_features = ['billing_ratio', 'billing_deviation', 'usage_diff', 
                'delay_risk_interaction', 'billing_per_unit', 'high_billing_flag']
for feat in new_features:
    corr = df[feat].corr(df['is_fraud'])
    correlations[feat] = float(corr)
    print(f"  -> {feat} <-> is_fraud: {corr:.3f}")

# Save correlation data for EDA
eda_data['correlations'] = correlations

# Define final feature columns
feature_columns = numeric_cols + [
    'billing_ratio', 'billing_deviation', 'usage_diff',
    'delay_risk_interaction', 'billing_per_unit', 'high_billing_flag',
    'payment_delay_category'
] + [f'{col}_missing' for col in feature_cols_for_flags]

print(f"\n  [SUCCESS] Total features: {len(feature_columns)}")
print(f"  -> Features: {feature_columns}")

# ============================================================
# STEP 6: TRAIN/TEST SPLIT
# ============================================================
print("\n[STEP 6] Train/Test Split...")

X = df[feature_columns].copy()
y = df['is_fraud'].astype(int)

# Replace any remaining inf/nan
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"  -> Training set: {X_train.shape[0]} samples")
print(f"  -> Test set: {X_test.shape[0]} samples")
print(f"  -> Train fraud rate: {y_train.mean()*100:.1f}%")
print(f"  -> Test fraud rate: {y_test.mean()*100:.1f}%")

# Save split info
eda_data['train_size'] = int(X_train.shape[0])
eda_data['test_size'] = int(X_test.shape[0])
eda_data['train_fraud_rate'] = float(y_train.mean() * 100)
eda_data['test_fraud_rate'] = float(y_test.mean() * 100)

# ============================================================
# STEP 7: MODEL TRAINING
# ============================================================
print("\n[STEP 7] Model Training...")

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
print("  -> Saved scaler")

# Define models
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE, C=1.0
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=8, min_samples_split=10, class_weight='balanced', random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    ),
}

if HAS_XGBOOST:
    # Calculate scale_pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    
    models['XGBoost'] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, eval_metric='logloss',
        use_label_encoder=False
    )

# Train all models and collect results
model_results = {}
best_model_name = None
best_recall = 0

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    # Use scaled data for Logistic Regression, raw for tree-based
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # ============================================================
    # STEP 8: MODEL EVALUATION
    # ============================================================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    
    # Precision-Recall curve data (Step 9)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    
    # Feature importances
    if name == 'Logistic Regression':
        importances = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_columns))
    
    importance_dict = {feature_columns[i]: float(importances[i]) 
                       for i in range(len(feature_columns))}
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    model_results[name] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'pr_curve': {
            'precision': pr_precision.tolist(), 
            'recall': pr_recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        },
        'feature_importances': importance_dict,
        'y_proba': y_proba.tolist(),
    }
    
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    ROC-AUC:   {roc_auc:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"      FN={cm[1][0]}  TP={cm[1][1]}")
    
    # Track best model by recall (primary) then F1 (secondary)
    if rec > best_recall or (rec == best_recall and f1 > model_results.get(best_model_name, {}).get('f1_score', 0)):
        best_recall = rec
        best_model_name = name

print(f"\n  [BEST] Best model: {best_model_name} (Recall={best_recall:.4f})")

# ============================================================
# STEP 9: THRESHOLD TUNING DATA
# ============================================================
print("\n[STEP 9] Threshold Tuning Analysis...")

# Use best model's probabilities for threshold analysis
best_proba = np.array(model_results[best_model_name]['y_proba'])
threshold_analysis = []
for threshold in np.arange(0.1, 1.0, 0.05):
    y_pred_thresh = (best_proba >= threshold).astype(int)
    if len(np.unique(y_pred_thresh)) < 2 and y_pred_thresh.sum() == 0:
        prec_t = 0
    else:
        prec_t = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec_t = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1_t = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    threshold_analysis.append({
        'threshold': round(float(threshold), 2),
        'precision': float(prec_t),
        'recall': float(rec_t),
        'f1_score': float(f1_t)
    })
    print(f"  -> Threshold {threshold:.2f}: Precision={prec_t:.3f}, Recall={rec_t:.3f}, F1={f1_t:.3f}")

# ============================================================
# SAVE ALL ARTIFACTS
# ============================================================
print("\n[SAVING] Model artifacts...")

# Save best model
best_model = models[best_model_name]
joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
print(f"  -> Saved best model ({best_model_name})")

# Save all models
for name, model in models.items():
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(model, os.path.join(MODEL_DIR, f'{safe_name}.pkl'))
print(f"  -> Saved all {len(models)} models")

# Save model results
joblib.dump(model_results, os.path.join(MODEL_DIR, 'model_results.pkl'))
print("  -> Saved model_results.pkl")

# Save feature names
joblib.dump(feature_columns, os.path.join(MODEL_DIR, 'feature_names.pkl'))
print("  -> Saved feature_names.pkl")

# Save medians for imputation in production
joblib.dump(medians, os.path.join(MODEL_DIR, 'medians.pkl'))
print("  -> Saved medians.pkl")

# Save EDA data
eda_data['threshold_analysis'] = threshold_analysis
eda_data['best_model_name'] = best_model_name
eda_data['feature_columns'] = feature_columns
eda_data['training_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
eda_data['n_features'] = len(feature_columns)
eda_data['models_trained'] = list(models.keys())

joblib.dump(eda_data, os.path.join(MODEL_DIR, 'eda_data.pkl'))
print("  -> Saved eda_data.pkl")

# Save the cleaned dataframe for the app's EDA visualizations
df_for_eda = df[numeric_cols + ['is_fraud', 'billing_ratio', 'billing_deviation', 
                                  'usage_diff', 'delay_risk_interaction', 
                                  'high_billing_flag']].copy()
joblib.dump(df_for_eda, os.path.join(MODEL_DIR, 'clean_data.pkl'))
print("  -> Saved clean_data.pkl")

# Save test data for the app
test_data = {
    'X_test': X_test,
    'y_test': y_test,
    'X_test_scaled': X_test_scaled,
}
joblib.dump(test_data, os.path.join(MODEL_DIR, 'test_data.pkl'))
print("  -> Saved test_data.pkl")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  TRAINING COMPLETE - FINAL SUMMARY")
print("=" * 70)
print(f"  Dataset: {df_raw.shape[0]} raw -> {df.shape[0]} clean rows")
print(f"  Features: {len(feature_columns)}")
print(f"  Models trained: {len(models)}")
print(f"  Best model: {best_model_name}")
print(f"  Best Recall: {model_results[best_model_name]['recall']:.4f}")
print(f"  Best F1: {model_results[best_model_name]['f1_score']:.4f}")
print(f"  Best ROC-AUC: {model_results[best_model_name]['roc_auc']:.4f}")
print(f"\n  Model comparison:")
for name, res in model_results.items():
    marker = " [BEST]" if name == best_model_name else ""
    print(f"    {name}: Acc={res['accuracy']:.3f} | Prec={res['precision']:.3f} | "
          f"Rec={res['recall']:.3f} | F1={res['f1_score']:.3f} | AUC={res['roc_auc']:.3f}{marker}")

print(f"\n  Artifacts saved to: {MODEL_DIR}")
print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
