from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data" / "raw" / "vidarbha_agri_350k.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "vidarbha_agri_cleaned.csv"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load raw dataset
# --------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Raw dataset not found: {INPUT_PATH}\n"
        "Run python src/generate_dataset.py first."
    )

df = pd.read_csv(INPUT_PATH)

print("=" * 65)
print("AGRICULTURAL DATA CLEANING")
print("=" * 65)

print("\nOriginal dataset shape:", df.shape)

# --------------------------------------------------
# 3. Remove duplicate records
# --------------------------------------------------

duplicate_count = df["record_id"].duplicated().sum()

print("\nDuplicate record IDs found:", duplicate_count)

df = df.drop_duplicates(subset=["record_id"], keep="first").copy()

print("Shape after duplicate removal:", df.shape)

# --------------------------------------------------
# 4. Standardize yield to quintals per hectare
# --------------------------------------------------
# Generator's known conventions:
# - Values above 100 represent kg/ha.
# - Values below 4 represent tonnes/ha.
# - Other values represent q/ha.
#
# These are synthetic-data heuristics, not universal
# agricultural unit-detection rules.

df["yield_unit_detected"] = "quintals_per_hectare"

kg_mask = df["yield_raw"] > 100
tonne_mask = df["yield_raw"] < 4

df.loc[kg_mask, "yield_unit_detected"] = "kilograms_per_hectare"
df.loc[tonne_mask, "yield_unit_detected"] = "tonnes_per_hectare"

df["yield_q_per_ha"] = df["yield_raw"].copy()

# Convert kg/ha to q/ha
df.loc[kg_mask, "yield_q_per_ha"] = (
    df.loc[kg_mask, "yield_raw"] / 100
)

# Convert tonnes/ha to q/ha
df.loc[tonne_mask, "yield_q_per_ha"] = (
    df.loc[tonne_mask, "yield_raw"] * 10
)

print("\nYield unit conversions:")
print(df["yield_unit_detected"].value_counts())

# --------------------------------------------------
# 5. Standardize mandi price to rupees per quintal
# --------------------------------------------------
# Generator's convention:
# Prices below 100 are assumed to be recorded per kg.
# Prices at or above 100 are assumed to be per quintal.

df["mandi_price_unit_detected"] = np.where(
    df["mandi_price_raw"] < 100,
    "rupees_per_kg",
    "rupees_per_quintal"
)

df["mandi_price_per_quintal"] = df["mandi_price_raw"].copy()

price_kg_mask = df["mandi_price_raw"] < 100

df.loc[price_kg_mask, "mandi_price_per_quintal"] = (
    df.loc[price_kg_mask, "mandi_price_raw"] * 100
)

print("\nMandi price unit conversions:")
print(df["mandi_price_unit_detected"].value_counts())

# --------------------------------------------------
# 6. Handle missing rainfall
# --------------------------------------------------
# Use the median rainfall for the same district.
# This uses predictor information, not the target.
# Record which values were imputed.

missing_before = df["lpa_rainfall_mm"].isna().sum()

df["rainfall_was_missing"] = (
    df["lpa_rainfall_mm"].isna().astype(int)
)

district_rainfall_medians = (
    df.groupby("district")["lpa_rainfall_mm"].median()
)

df["lpa_rainfall_mm"] = df["lpa_rainfall_mm"].fillna(
    df["district"].map(district_rainfall_medians)
)

# Fallback in case a district has no observed rainfall.
overall_rainfall_median = df["lpa_rainfall_mm"].median()

df["lpa_rainfall_mm"] = df["lpa_rainfall_mm"].fillna(
    overall_rainfall_median
)

missing_after = df["lpa_rainfall_mm"].isna().sum()

print("\nRainfall missing before:", missing_before)
print("Rainfall missing after:", missing_after)

# --------------------------------------------------
# 7. Validate numerical values
# --------------------------------------------------

numeric_columns = [
    "area_hectares",
    "lpa_rainfall_mm",
    "cum_rainfall_cutoff_mm",
    "dry_spell_max_days",
    "heavy_rain_days",
    "soil_nitrogen",
    "soil_phosphorus",
    "soil_potassium",
    "soil_organic_carbon",
    "soil_ph",
    "msp_quintal",
    "yield_q_per_ha",
    "mandi_price_per_quintal"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Flag impossible or suspicious values.
df["invalid_area_flag"] = (
    df["area_hectares"] <= 0
).astype(int)

df["invalid_yield_flag"] = (
    (df["yield_q_per_ha"] <= 0)
    | (df["yield_q_per_ha"] > 100)
).astype(int)

df["invalid_price_flag"] = (
    df["mandi_price_per_quintal"] <= 0
).astype(int)

df["invalid_rainfall_flag"] = (
    (df["lpa_rainfall_mm"] < 0)
    | (df["cum_rainfall_cutoff_mm"] < 0)
).astype(int)

print("\nInvalid-value flags:")
for column in [
    "invalid_area_flag",
    "invalid_yield_flag",
    "invalid_price_flag",
    "invalid_rainfall_flag"
]:
    print(f"{column}: {df[column].sum()}")

# --------------------------------------------------
# 8. Create IQR outlier flags
# --------------------------------------------------
# We flag outliers for investigation.
# We do not automatically delete them.

outlier_columns = [
    "yield_q_per_ha",
    "mandi_price_per_quintal",
    "lpa_rainfall_mm",
    "soil_nitrogen",
    "soil_phosphorus",
    "soil_potassium",
    "soil_organic_carbon"
]

for column in outlier_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    flag_column = f"{column}_outlier_flag"

    df[flag_column] = (
        (df[column] < lower_bound)
        | (df[column] > upper_bound)
    ).astype(int)

    print(
        f"{column}: {df[flag_column].sum()} potential outliers"
    )

# --------------------------------------------------
# 9. Remove post-harvest leakage column
# --------------------------------------------------

leakage_column = "post_harvest_mandi_arrivals_tonnes"

if leakage_column in df.columns:
    df = df.drop(columns=[leakage_column])

print("\nRemoved leakage column:", leakage_column)

# --------------------------------------------------
# 10. Check remaining missing values
# --------------------------------------------------

print("\nRemaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# --------------------------------------------------
# 11. Save cleaned dataset
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 65)
print("DATA CLEANING COMPLETED")
print("=" * 65)

print("Final dataset shape:", df.shape)
print("Output file:", OUTPUT_PATH)

print("\nStandardized yield summary:")
print(df["yield_q_per_ha"].describe().round(2))

print("\nStandardized mandi price summary:")
print(df["mandi_price_per_quintal"].describe().round(2))

print("\nCleaned dataset saved successfully.")