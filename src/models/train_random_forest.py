from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"


# -------------------------
# Loading data
# -------------------------

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)


# -------------------------
# Splitting X and y
# -------------------------

X_train = train_df.drop("default", axis=1)
y_train = train_df["default"]

X_validation = validation_df.drop("default", axis=1)
y_validation = validation_df["default"]


print("Train shape:", X_train.shape)
print("Validation shape:", X_validation.shape)


# -------------------------
# Model
# -------------------------

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# -------------------------
# Cross-validation
# -------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("\n--- CROSS VALIDATION ---")
print("ROC-AUC scores:", cv_scores)
print(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
print(f"Std ROC-AUC:  {cv_scores.std():.4f}")

# -------------------------
# Training
# -------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")


# -------------------------
# Prediction
# -------------------------

y_probability = model.predict_proba(X_validation)[:, 1]

threshold = 0.5

y_prediction = (y_probability >= threshold).astype(int)


# -------------------------
# Evaluation
# -------------------------

accuracy = accuracy_score(y_validation, y_prediction)
precision = precision_score(y_validation, y_prediction)
recall = recall_score(y_validation, y_prediction)
f1 = f1_score(y_validation, y_prediction)
roc_auc = roc_auc_score(y_validation, y_probability)

conf_matrix = confusion_matrix(
    y_validation,
    y_prediction
)


print("\n--- RANDOM FOREST VALIDATION RESULTS ---")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion matrix:")
print(conf_matrix)


# -------------------------
# Feature importance
# -------------------------

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n--- FEATURE IMPORTANCE ---")
print(feature_importance)

# -------------------------
# Feature importance plot
# -------------------------

plt.figure(figsize=(10, 8))

plt.barh(
    feature_importance["feature"],
    feature_importance["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.show()