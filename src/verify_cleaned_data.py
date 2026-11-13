from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    ROOT / "data" / "processed" / "vidarbha_agri_cleaned.csv"
)

df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("CLEANED DATASET VERIFICATION")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nDuplicate record IDs:")
print(df["record_id"].duplicated().sum())

print("\nMissing rainfall values:")
print(df["lpa_rainfall_mm"].isna().sum())

print("\nYield unit distribution:")
print(df["yield_unit_detected"].value_counts())

print("\nMandi price unit distribution:")
print(df["mandi_price_unit_detected"].value_counts())

print("\nStandardized yield statistics:")
print(df["yield_q_per_ha"].describe())

print("\nStandardized mandi price statistics:")
print(df["mandi_price_per_quintal"].describe())

print("\nLeakage column present?")
print("post_harvest_mandi_arrivals_tonnes" in df.columns)

print("\nRemaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nCLEANED DATA VERIFICATION COMPLETED!")
