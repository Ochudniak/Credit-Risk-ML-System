from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "clean_credit_data.csv"

TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"


if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Raw data file not found: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

print("Clean dataset loaded successfully.")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Target values: {sorted(df['default'].unique())}")

X = df.drop('default', axis=1)
Y = df['default']

print(f"\nX shape: {X.shape}")
print(f"y shape: {Y.shape}")

#dzielimy dane w proporcjach 70%/30%
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    Y,
    test_size=0.30,
    random_state=42,
    stratify=Y
)

#dzielimy 30% na dwie równe części validation i test
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

train_df = pd.concat([X_train, y_train], axis=1)
validation_df = pd.concat([X_validation, y_validation], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv(TRAIN_PATH, index=False)
validation_df.to_csv(VALIDATION_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("\nData split completed successfully.")

print(f"\nTrain:      {train_df.shape}")
print(f"Validation: {validation_df.shape}")
print(f"Test:       {test_df.shape}")

print("\nDefault class distribution:")

print("\nTrain:")
print(train_df["default"].value_counts(normalize=True))

print("\nValidation:")
print(validation_df["default"].value_counts(normalize=True))

print("\nTest:")
print(test_df["default"].value_counts(normalize=True))

print("\nFiles saved:")
print(f"Train:      {TRAIN_PATH}")
print(f"Validation: {VALIDATION_PATH}")
print(f"Test:       {TEST_PATH}")