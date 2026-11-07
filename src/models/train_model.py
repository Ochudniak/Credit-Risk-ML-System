import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_ROOT = PROJECT_ROOT / "data" / "processed" / "validation.csv"

train_df = pd.read_csv(TRAIN_ROOT)
validation_df = pd.read_csv(VALIDATION_ROOT)

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {validation_df.shape}")

X_train = train_df.drop('default', axis=1)
y_train = train_df['default']

X_validation = validation_df.drop('default', axis=1)
y_validation = validation_df['default']

print(f"{X_train.shape}")
print(f"{y_train.shape}")
print(f"{X_validation.shape}")
print(f"{y_validation.shape}")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_validation)

cm = confusion_matrix(y_validation, y_pred)

y_proba = model.predict_proba(X_validation)

for y in y_proba:
    if y[:, 0] > 30:
            Y


print(f"{y_pred[:10]}")
print(np.unique(y_pred, return_counts=True))
print(cm)
print(f"Precision score: {precision_score(y_validation, y_pred)}")
print(f"Recall score: {recall_score(y_validation, y_pred)}")