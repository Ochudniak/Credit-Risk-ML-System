from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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

validation_df = pd.read_csv(VALIDATION_PATH)

X_validation = validation_df.drop("default", axis=1)
y_validation = validation_df["default"]


train_df = pd.read_csv(TRAIN_PATH)

X_train = train_df.drop("default", axis=1)
y_train = train_df["default"]


model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


param_grid = {
    "n_estimators": [100, 200, 300, 500, 700],
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2"]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2
)


search.fit(X_train, y_train)
best_model = search.best_estimator_

y_probability = best_model.predict_proba(X_validation)[:, 1]
threshold = 0.5

y_prediction = (y_probability >= threshold).astype(int)

accuracy = accuracy_score(y_validation, y_prediction)
precision = precision_score(y_validation, y_prediction)
recall = recall_score(y_validation, y_prediction)
f1 = f1_score(y_validation, y_prediction)

roc_auc = roc_auc_score(
    y_validation,
    y_probability
)

conf_matrix = confusion_matrix(
    y_validation,
    y_prediction
)


print("\n--- BEST PARAMETERS ---")
print(search.best_params_)

print("\n--- BEST CROSS-VALIDATION SCORE ---")
print(f"ROC-AUC: {search.best_score_:.4f}")

print("\n--- TUNED RANDOM FOREST VALIDATION ---")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion matrix:")
print(conf_matrix)