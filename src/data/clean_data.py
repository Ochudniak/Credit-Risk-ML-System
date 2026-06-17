from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "default_of_credit_card_clients.xls"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DATA_DIR / "clean_credit_data.csv"


if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")

df = pd.read_excel(RAW_DATA_PATH, header=1)

print("Raw dataset loaded successfully.")

df = df.drop('ID', axis=1)  #Deleting ID column
df.rename(columns={'default payment next month': 'default'}, inplace=True)

#Fixing unknown values in EDUCATION column
df.loc[df['EDUCATION'] == 0, 'EDUCATION'] = 4
df.loc[df['EDUCATION'] == 5, 'EDUCATION'] = 4
df.loc[df['EDUCATION'] == 6, 'EDUCATION'] = 4

#Fixing unknown values in MARRIAGE column
df.loc[df['MARRIAGE'] == 0, 'MARRIAGE'] = 3

#Checking output in terminal
print(f"Sex values: {df['SEX'].unique()}")
print(f"Marriage values: {df['MARRIAGE'].unique()}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

#Creating new file
df.to_csv(OUTPUT_PATH, index=False)
