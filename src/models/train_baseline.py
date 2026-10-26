from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
# Model pipeline
# -------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])


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


print("\n--- VALIDATION RESULTS ---")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion matrix:")
print(conf_matrix)

# -------------------------
# Threshold analysis
# -------------------------

precision_values, recall_values, thresholds = precision_recall_curve(
    y_validation,
    y_probability
)

plt.figure(figsize=(10, 6))

plt.plot(
    thresholds,
    precision_values[:-1],
    label="Precision"
)

plt.plot(
    thresholds,
    recall_values[:-1],
    label="Recall"
)

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")

plt.legend()
plt.grid()

plt.show()