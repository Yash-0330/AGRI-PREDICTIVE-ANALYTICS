from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT / "data" / "processed" / "vidarbha_agri_engineered.csv"
)

SPLIT_DIR = ROOT / "data" / "processed" / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load engineered dataset
# --------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Engineered dataset not found: {INPUT_PATH}\n"
        "Run python src/feature_engineering.py first."
    )

df = pd.read_csv(INPUT_PATH)

print("=" * 65)
print("TIME-BASED DATASET SPLITTING")
print("=" * 65)

print("\nOriginal dataset shape:", df.shape)

# --------------------------------------------------
# 3. Validate required columns
# --------------------------------------------------

required_columns = [
    "year",
    "district",
    "crop",
    "yield_q_per_ha"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# --------------------------------------------------
# 4. Validate target variable
# --------------------------------------------------

df["yield_q_per_ha"] = pd.to_numeric(
    df["yield_q_per_ha"],
    errors="coerce"
)

target_missing = df["yield_q_per_ha"].isna().sum()

print("\nMissing target values:", target_missing)

# Remove rows without a usable target.
df = df.dropna(subset=["yield_q_per_ha"]).copy()

# Exclude clearly invalid target values.
# The threshold is a project-level validation rule,
# not a universal agricultural yield limit.
invalid_target_mask = (
    (df["yield_q_per_ha"] <= 0)
    | (df["yield_q_per_ha"] > 100)
)

print(
    "Rows with invalid target values:",
    invalid_target_mask.sum()
)

df = df.loc[~invalid_target_mask].copy()

# --------------------------------------------------
# 5. Set chronological periods
# --------------------------------------------------

train = df[df["year"].between(2015, 2022)].copy()

validation = df[df["year"] == 2023].copy()

test = df[df["year"] == 2024].copy()

# --------------------------------------------------
# 6. Check split completeness
# --------------------------------------------------

if train.empty:
    raise ValueError("Training dataset is empty.")

if validation.empty:
    raise ValueError("Validation dataset is empty.")

if test.empty:
    raise ValueError("Test dataset is empty.")

unexpected_years = sorted(
    set(df["year"].dropna().unique())
    - set(range(2015, 2025))
)

if unexpected_years:
    print("\nWarning: unexpected years found:", unexpected_years)

# --------------------------------------------------
# 7. Verify no year overlap
# --------------------------------------------------

train_years = set(train["year"].unique())
validation_years = set(validation["year"].unique())
test_years = set(test["year"].unique())

assert train_years.isdisjoint(validation_years)
assert train_years.isdisjoint(test_years)
assert validation_years.isdisjoint(test_years)

assert train["year"].max() < validation["year"].min()
assert validation["year"].max() < test["year"].min()

# --------------------------------------------------
# 8. Sort each dataset
# --------------------------------------------------

sort_columns = ["year", "district", "crop"]

train = train.sort_values(sort_columns).reset_index(drop=True)
validation = validation.sort_values(sort_columns).reset_index(drop=True)
test = test.sort_values(sort_columns).reset_index(drop=True)

# --------------------------------------------------
# 9. Save split datasets
# --------------------------------------------------

train_path = SPLIT_DIR / "train.csv"
validation_path = SPLIT_DIR / "validation.csv"
test_path = SPLIT_DIR / "test.csv"

train.to_csv(train_path, index=False)
validation.to_csv(validation_path, index=False)
test.to_csv(test_path, index=False)

# --------------------------------------------------
# 10. Display split summary
# --------------------------------------------------

print("\n" + "=" * 65)
print("DATA SPLIT SUMMARY")
print("=" * 65)

summary = pd.DataFrame({
    "Dataset": ["Training", "Validation", "Testing"],
    "Years": [
        "2015–2022",
        "2023",
        "2024"
    ],
    "Rows": [
        len(train),
        len(validation),
        len(test)
    ],
    "Percentage": [
        round(len(train) / len(df) * 100, 2),
        round(len(validation) / len(df) * 100, 2),
        round(len(test) / len(df) * 100, 2)
    ]
})

print(summary.to_string(index=False))

print("\nTraining years:", sorted(train_years))
print("Validation years:", sorted(validation_years))
print("Testing years:", sorted(test_years))

print("\nDistricts in training:", train["district"].nunique())
print("Crops in training:", train["crop"].nunique())

print("\nTarget averages:")
print("Training:", round(train["yield_q_per_ha"].mean(), 2))
print("Validation:", round(validation["yield_q_per_ha"].mean(), 2))
print("Testing:", round(test["yield_q_per_ha"].mean(), 2))

print("\nSaved files:")
print(train_path)
print(validation_path)
print(test_path)

print("\n" + "=" * 65)
print("TIME-BASED SPLITTING COMPLETED!")
print("=" * 65)
