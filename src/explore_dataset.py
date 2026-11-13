from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "vidarbha_agri_350k.csv"

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run: python src/generate_dataset.py"
    )

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 3. Basic dataset information
# --------------------------------------------------

print("\n" + "=" * 65)
print("AGRI PREDICTIVE ANALYTICS - DATASET EXPLORATION")
print("=" * 65)

print("\n1. DATASET DIMENSIONS")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\n2. FIRST FIVE RECORDS")
print(df.head().to_string(index=False))

print("\n3. LAST FIVE RECORDS")
print(df.tail().to_string(index=False))

print("\n4. COLUMN NAMES")
for number, column in enumerate(df.columns, start=1):
    print(f"{number:2}. {column}")

print("\n5. DATA TYPES")
print(df.dtypes)

print("\n6. DATASET INFORMATION")
df.info()

# --------------------------------------------------
# 4. Missing-value analysis
# --------------------------------------------------

print("\n7. MISSING VALUES")
missing = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (df.isnull().mean() * 100).round(2)
})

print(missing[missing["Missing_Count"] > 0].sort_values(
    by="Missing_Percentage", ascending=False
))

# --------------------------------------------------
# 5. Duplicate analysis
# --------------------------------------------------

print("\n8. DUPLICATE ANALYSIS")
print("Completely duplicated rows:", df.duplicated().sum())
print("Duplicate record IDs:", df["record_id"].duplicated().sum())

# --------------------------------------------------
# 6. Category analysis
# --------------------------------------------------

print("\n9. DISTRICT DISTRIBUTION")
print(df["district"].value_counts())

print("\n10. CROP DISTRIBUTION")
print(df["crop"].value_counts())

print("\n11. SEASON DISTRIBUTION")
print(df["season"].value_counts())

print("\n12. YEAR DISTRIBUTION")
print(df["year"].value_counts().sort_index())

# --------------------------------------------------
# 7. Numerical statistical summary
# --------------------------------------------------

print("\n13. NUMERICAL SUMMARY")
print(df.describe().round(2).to_string())

# --------------------------------------------------
# 8. Check unusual unit values
# --------------------------------------------------

print("\n14. RAW YIELD VALUE CHECK")
print(df["yield_raw"].describe().round(2))

print("\n15. RAW MANDI PRICE CHECK")
print(df["mandi_price_raw"].describe().round(2))

# --------------------------------------------------
# 9. Check target and leakage-related column
# --------------------------------------------------

print("\n16. TARGET COLUMN")
print("Target candidate: yield_raw")
print("Missing target values:", df["yield_raw"].isnull().sum())

print("\n17. POSSIBLE LEAKAGE COLUMN")
print("post_harvest_mandi_arrivals_tonnes")
print("This column represents post-harvest information.")
print("We will exclude it from predictive model features.")

# --------------------------------------------------
# 10. Save exploration summary
# --------------------------------------------------

report_path = ROOT / "reports" / "dataset_exploration_summary.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("AGRI DATASET EXPLORATION SUMMARY\n")
    file.write("=" * 50 + "\n")
    file.write(f"Rows: {df.shape[0]:,}\n")
    file.write(f"Columns: {df.shape[1]}\n\n")

    file.write("COLUMN DATA TYPES\n")
    file.write(df.dtypes.to_string())
    file.write("\n\nMISSING VALUES\n")
    file.write(missing.to_string())
    file.write("\n\nDISTRICT DISTRIBUTION\n")
    file.write(df["district"].value_counts().to_string())
    file.write("\n\nCROP DISTRIBUTION\n")
    file.write(df["crop"].value_counts().to_string())
    file.write("\n\nNUMERICAL SUMMARY\n")
    file.write(df.describe().round(2).to_string())

print("\n" + "=" * 65)
print("DATASET EXPLORATION COMPLETED!")
print(f"Summary saved at: {report_path}")
print("=" * 65)