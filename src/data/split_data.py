from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "clean_credit_data.csv"

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Raw data file not found: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

print("Clean dataset loaded successfully.")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Target values: {sorted(df['default'].unique())}")

X = df.drop('default', axis=1)
Y = df['default']

print(f"X: {X.shape}")
print(f"Y: {Y.shape}")