import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

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

depths = [4, 6, 8, 10, 12]
ras_max = 0

# szukanie optymalnej głebokosci drzewa

for depth in depths:
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=depth)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_validation)
    y_proba_t = model.predict_proba(X_train)

    ras = roc_auc_score(y_validation, y_proba[:,1])
    ras_t = roc_auc_score(y_train, y_proba_t[:,1])

    print(f"Dla depth = {depth}")
    print(f"RAS walidacji {ras}")
    print(f"RAS treningu: {ras_t}")
    print("________________________")

    if ras_max < ras:
        ras_max = ras
        depth_best = depth

leafs = [1, 3, 5, 10, 20]
ras_max = 0

# szukanie optymalnej ilosci lisci

for leaf in leafs:
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=depth_best, min_samples_leaf=leaf)
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_validation)
    y_proba_t = model.predict_proba(X_train)
    
    ras = roc_auc_score(y_validation, y_proba[:,1])
    ras_t = roc_auc_score(y_train, y_proba_t[:,1])
    
    print(f"Dla leaf = {leaf}")
    print(f"RAS walidacji {ras}")
    print(f"RAS treningu: {ras_t}")
    print("________________________")
    
    if ras_max < ras:
        ras_max = ras
        leaf_best = leaf


model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=depth_best)
model.fit(X_train, y_train)

y_pred = model.predict(X_validation)
cm = confusion_matrix(y_validation, y_pred)

#print(f"{y_pred[:10]}")
#print(np.unique(y_pred, return_counts=True))
#print(cm)
#print(f"Precision score: {precision_score(y_validation, y_pred)}")
#print(f"Recall score: {recall_score(y_validation, y_pred)}")