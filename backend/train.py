# ============================================================
# CREDIT CARD FRAUD DETECTION
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION")
print("=" * 60)

dataset_path = Path(__file__).resolve().parent.parent / "credit_card_fraud.csv"
df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. UNDERSTAND DATASET
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nCreating EDA graphs...")

# Class distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    x="Class",
    data=df
)

plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.show()


# Amount distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=50
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()


# Correlation heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title("Feature Correlation")
plt.show()


# ============================================================
# 4. REMOVE UNNECESSARY COLUMN
# ============================================================

X = df.drop(
    columns=["Class", "TransactionID"]
)

y = df["Class"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Class")


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# Save scaler
joblib.dump(
    scaler,
    "scaler.pkl"
)

print("\nScaler saved as scaler.pkl")


# ============================================================
# 7. MODEL 1 - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_prediction = logistic_model.predict(
    X_test_scaled
)

logistic_probability = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]

print("Logistic Regression training completed!")


# ============================================================
# 8. MODEL 2 - RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

random_forest_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Random Forest does not require scaled data
random_forest_model.fit(
    X_train,
    y_train
)

random_forest_prediction = random_forest_model.predict(
    X_test
)

random_forest_probability = random_forest_model.predict_proba(
    X_test
)[:, 1]

print("Random Forest training completed!")


# ============================================================
# 9. MODEL 3 - XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_prediction = xgb_model.predict(
    X_test
)

xgb_probability = xgb_model.predict_proba(
    X_test
)[:, 1]

print("XGBoost training completed!")


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_actual,
    y_prediction,
    probability
):

    accuracy = accuracy_score(
        y_actual,
        y_prediction
    )

    precision = precision_score(
        y_actual,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_actual,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_actual,
        y_prediction,
        zero_division=0
    )

    auc = roc_auc_score(
        y_actual,
        probability
    )

    print("\n" + "-" * 50)
    print(model_name)
    print("-" * 50)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_actual,
            y_prediction,
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    }


# ============================================================
# 11. EVALUATE ALL MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

results = []

# Logistic Regression
results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_prediction,
        logistic_probability
    )
)

# Random Forest
results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        random_forest_prediction,
        random_forest_probability
    )
)

# XGBoost
results.append(
    evaluate_model(
        "XGBoost",
        y_test,
        xgb_prediction,
        xgb_probability
    )
)


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 13. COMPARISON GRAPH
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

for metric in metrics:

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=results_df,
        x="Model",
        y=metric
    )

    plt.title(
        metric + " Comparison"
    )

    plt.ylim(0, 1)

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 14. CONFUSION MATRICES
# ============================================================

models_predictions = [
    (
        "Logistic Regression",
        logistic_prediction
    ),
    (
        "Random Forest",
        random_forest_prediction
    ),
    (
        "XGBoost",
        xgb_prediction
    )
]

for model_name, prediction in models_predictions:

    cm = confusion_matrix(
        y_test,
        prediction
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        model_name + " Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()


# ============================================================
# 15. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = random_forest_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance)


plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.show()


# ============================================================
# 16. SAVE RANDOM FOREST MODEL
# ============================================================

joblib.dump(
    random_forest_model,
    "model.pkl"
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("Random Forest model saved as: model.pkl")
print("Scaler saved as: scaler.pkl")


# ============================================================
# 17. TEST NEW TRANSACTION
# ============================================================

print("\n" + "=" * 60)
print("NEW TRANSACTION PREDICTION")
print("=" * 60)

new_transaction = pd.DataFrame([{

    "Time": 50000,

    "V1": 0.5,
    "V2": 0.2,
    "V3": 0.8,
    "V4": -0.3,
    "V5": 0.4,

    "V6": -0.2,
    "V7": 0.6,
    "V8": 0.1,
    "V9": 0.3,
    "V10": -0.1,

    "Amount": 50.00

}])


# Random Forest prediction
new_prediction = random_forest_model.predict(
    new_transaction
)

new_probability = random_forest_model.predict_proba(
    new_transaction
)

fraud_probability = (
    new_probability[0][1] * 100
)


print("\nTransaction Details:")
print(new_transaction)

print(
    "\nFraud Probability:",
    round(fraud_probability, 2),
    "%"
)


if new_prediction[0] == 1:

    print("\nRESULT: FRAUDULENT TRANSACTION")

else:

    print("\nRESULT: LEGITIMATE TRANSACTION")


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("ML PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)