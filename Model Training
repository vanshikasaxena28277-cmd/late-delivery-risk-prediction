# ============================================================
# Late Delivery Risk Prediction - ML Model Training
# APL Logistics Supply Chain Project
# ============================================================
# HOW TO RUN:
#   1. Download dataset from Kaggle: "DataCo Supply Chain Dataset"
#      (file name: DataCoSupplyChainDataset.csv)
#   2. Place the CSV in the same folder as this script
#   3. Run: python model_training.py
# ============================================================
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
import joblib
 
 
# ============================================================
# STEP 1 - LOAD DATA
# ============================================================
print("=" * 55)
print("STEP 1: Loading Dataset")
print("=" * 55)
 
try:
    df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin-1')
    print(f"  Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
except FileNotFoundError:
    print("\n  ERROR: DataCoSupplyChainDataset.csv not found!")
    print("  Please download from Kaggle and place in this folder.")
    exit()
 
 
# ============================================================
# STEP 2 - DATA CLEANING
# ============================================================
print("\nSTEP 2: Cleaning Data")
 
# Drop columns that won't help prediction or leak the label
useless_cols = [
    'Customer Fname', 'Customer Lname', 'Customer Street',
    'Customer Zipcode', 'Customer Id', 'Order Customer Id',
    'Product Name', 'Delivery Status',
    'order date (DateOrders)', 'shipping date (DateOrders)'
]
useless_cols = [c for c in useless_cols if c in df.columns]
df.drop(columns=useless_cols, inplace=True)
 
# Remove rows where target is missing
df.dropna(subset=['Late_delivery_risk'], inplace=True)
 
# Fill numeric missing values with median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)
 
# Fill text missing values with most common value
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
 
print(f"  Clean data shape: {df.shape}")
 
 
# ============================================================
# STEP 3 - FEATURE ENGINEERING
# ============================================================
print("\nSTEP 3: Engineering New Features")
 
# Gap between real and scheduled shipping days
if ('Days for shipping (real)' in df.columns and
        'Days for shipment (scheduled)' in df.columns):
    df['shipping_delay_gap'] = (
        df['Days for shipping (real)'] - df['Days for shipment (scheduled)']
    )
    df['shipping_pressure_ratio'] = (
        df['Days for shipment (scheduled)'] /
        (df['Days for shipping (real)'] + 1)
    )
    print("  + shipping_delay_gap")
    print("  + shipping_pressure_ratio")
 
# Order complexity
if ('Order Item Quantity' in df.columns and
        'Order Item Discount Rate' in df.columns):
    df['order_complexity'] = (
        df['Order Item Quantity'] * df['Order Item Discount Rate']
    )
    print("  + order_complexity")
 
# Flag orders with negative profit (risky orders)
if 'Order Item Profit Ratio' in df.columns:
    df['low_profit_flag'] = (df['Order Item Profit Ratio'] < 0).astype(int)
    print("  + low_profit_flag")
 
# High discount flag
if 'Order Item Discount Rate' in df.columns:
    df['high_discount_flag'] = (df['Order Item Discount Rate'] > 0.2).astype(int)
    print("  + high_discount_flag")
 
 
# ============================================================
# STEP 4 - ENCODE CATEGORICAL COLUMNS
# ============================================================
print("\nSTEP 4: Encoding Categorical Variables")
 
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"  Encoding {len(cat_cols)} categorical columns")
 
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
 
 
# ============================================================
# STEP 5 - PREPARE X AND y
# ============================================================
print("\nSTEP 5: Preparing Features and Target")
 
target = 'Late_delivery_risk'
X = df.drop(columns=[target])
y = df[target]
 
print(f"  Target distribution:")
vc = y.value_counts()
for val, cnt in vc.items():
    label = "On Time" if val == 0 else "Late"
    print(f"    {label} ({val}): {cnt:,} ({cnt/len(y)*100:.1f}%)")
 
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
 
# Save for Streamlit app
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
print("  Saved: scaler.pkl, feature_names.pkl")
 
 
# ============================================================
# STEP 6 - TRAIN/TEST SPLIT + SMOTE
# ============================================================
print("\nSTEP 6: Train/Test Split and SMOTE")
 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
 
try:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE - Train: {len(X_train_bal):,}")
except Exception as e:
    print(f"  SMOTE skipped ({e}), using raw train data")
    X_train_bal, y_train_bal = X_train, y_train
 
 
# ============================================================
# STEP 7 - TRAIN MODELS
# ============================================================
print("\nSTEP 7: Training Models")
 
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    )
}
 
results = {}
for name, model in models.items():
    print(f"  Training {name}...", end=" ")
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
 
    results[name] = {
        'model': model,
        'auc': roc_auc_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    print(f"Done | AUC={results[name]['auc']:.3f} F1={results[name]['f1']:.3f}")
 
 
# ============================================================
# STEP 8 - SAVE BEST MODEL
# ============================================================
print("\nSTEP 8: Saving Best Model")
 
best_name = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_name]['model']
joblib.dump(best_model, 'best_model.pkl')
 
# Save feature importance for Streamlit
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=X.columns)
    joblib.dump(imp.nlargest(15), 'top_features.pkl')
 
print(f"  Best model: {best_name} (AUC = {results[best_name]['auc']:.4f})")
print("  Saved: best_model.pkl, top_features.pkl")
 
 
# ============================================================
# STEP 9 - SAVE PLOTS
# ============================================================
print("\nSTEP 9: Generating and Saving Plots")
os.makedirs('plots', exist_ok=True)
 
# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices - All Models', fontsize=14)
for i, (name, r) in enumerate(results.items()):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=axes[i], xticklabels=['On Time', 'Late'],
                yticklabels=['On Time', 'Late'])
    axes[i].set_title(f'{name}\nAUC={r["auc"]:.3f}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('plots/confusion_matrices.png', dpi=150)
plt.close()
 
# ROC curves
plt.figure(figsize=(8, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Model Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('plots/roc_curves.png', dpi=150)
plt.close()
 
# Feature importance
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=X.columns)
    top15 = imp.nlargest(15).sort_values()
    plt.figure(figsize=(10, 6))
    top15.plot(kind='barh', color='steelblue')
    plt.title(f'Top 15 Important Features - {best_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=150)
    plt.close()
 
# Model comparison
comp_df = pd.DataFrame(
    {n: [r['auc'], r['precision'], r['recall'], r['f1']]
     for n, r in results.items()},
    index=['AUC', 'Precision', 'Recall', 'F1 Score']
)
comp_df.T.plot(kind='bar', figsize=(10, 5))
plt.title('Model Performance Comparison')
plt.ylabel('Score (0 to 1)')
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150)
plt.close()
 
print("  Saved all plots in /plots folder")
 
 
# ============================================================
# STEP 10 - PRINT SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("=" * 55)
print(f"{'Model':<25} {'AUC':>6} {'Precision':>10} {'Recall':>8} {'F1':>6}")
print("-" * 55)
for name, r in results.items():
    print(f"{name:<25} {r['auc']:>6.3f} {r['precision']:>10.3f} "
          f"{r['recall']:>8.3f} {r['f1']:>6.3f}")
print(f"\nBest Model: {best_name}")
print("\nFiles ready:")
print("  best_model.pkl, scaler.pkl, feature_names.pkl, top_features.pkl")
print("  plots/confusion_matrices.png")
print("  plots/roc_curves.png")
print("  plots/feature_importance.png")
print("  plots/model_comparison.png")
print("\nNow run the dashboard: streamlit run app.py")
