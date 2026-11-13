
import pandas as pd
from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parent.parent

# Dataset path
file_path = ROOT / "data" / "raw" / "vidarbha_agri_350k.csv"

# Load dataset
df = pd.read_csv(file_path)

print("=" * 55)
print("AGRI DATASET VERIFICATION")
print("=" * 55)

# 1. Dataset dimensions
print("\n1. Dataset Shape:")
print(df.shape)

# 2. First five records
print("\n2. First Five Records:")
print(df.head())

# 3. Column names
print("\n3. Column Names:")
print(df.columns.tolist())

# 4. Missing values
print("\n4. Missing Values:")
print(df.isnull().sum())

# 5. District distribution
print("\n5. District Distribution:")
print(df["district"].value_counts())

# 6. Crop distribution
print("\n6. Crop Distribution:")
print(df["crop"].value_counts())

# 7. Duplicate records
print("\n7. Duplicate Record IDs:")
print(df["record_id"].duplicated().sum())

# 8. Statistical summary
print("\n8. Statistical Summary:")
print(df.describe())

print("\nDATASET VERIFICATION COMPLETED!")
