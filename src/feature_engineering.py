from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT / "data" / "processed" / "vidarbha_agri_cleaned.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "vidarbha_agri_engineered.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {INPUT_PATH}\n"
        "Run python src/data_cleaning.py first."
    )

df = pd.read_csv(INPUT_PATH)

print("=" * 65)
print("AGRICULTURAL FEATURE ENGINEERING")
print("=" * 65)

print("\nInput dataset shape:", df.shape)

# --------------------------------------------------
# 3. Restore originally missing rainfall values
# --------------------------------------------------
# Step 4 filled missing rainfall for exploration.
# Restore the original missing values so that the
# final modelling pipeline can impute them using
# training data only.

if "rainfall_was_missing" in df.columns:
    original_missing_mask = df["rainfall_was_missing"].eq(1)
    df.loc[original_missing_mask, "lpa_rainfall_mm"] = np.nan

print(
    "\nRainfall values restored to missing:",
    df["lpa_rainfall_mm"].isna().sum()
)

# --------------------------------------------------
# 4. Rainfall-related features
# --------------------------------------------------

# Rainfall relative to district long-term average.
df["rainfall_adequacy_ratio"] = (
    df["cum_rainfall_cutoff_mm"] / df["lpa_rainfall_mm"]
)

# Difference between observed cumulative rainfall
# and the long-term average.
df["rainfall_difference_mm"] = (
    df["cum_rainfall_cutoff_mm"] - df["lpa_rainfall_mm"]
)

# Rainfall per dry-spell day, with protection
# against division by zero.
df["rainfall_per_dry_spell_day"] = (
    df["cum_rainfall_cutoff_mm"]
    / (df["dry_spell_max_days"] + 1)
)

# A simple weather-stress indicator.
df["weather_stress_index"] = (
    df["dry_spell_max_days"] * 0.6
    + df["heavy_rain_days"] * 0.4
)

# --------------------------------------------------
# 5. Soil-related features
# --------------------------------------------------

df["total_npk"] = (
    df["soil_nitrogen"]
    + df["soil_phosphorus"]
    + df["soil_potassium"]
)

df["nitrogen_phosphorus_ratio"] = (
    df["soil_nitrogen"]
    / (df["soil_phosphorus"] + 1)
)

df["potassium_nitrogen_ratio"] = (
    df["soil_potassium"]
    / (df["soil_nitrogen"] + 1)
)

# Distance from a neutral-ish pH reference.
# This is a generic indicator, not a crop-specific
# agronomic suitability score.
df["soil_ph_distance_from_6_5"] = (
    df["soil_ph"] - 6.5
).abs()

# --------------------------------------------------
# 6. Market-related features
# --------------------------------------------------

df["price_to_msp_ratio"] = (
    df["mandi_price_per_quintal"] / df["msp_quintal"]
)

df["price_premium_over_msp"] = (
    df["mandi_price_per_quintal"] - df["msp_quintal"]
)

# NOTE:
# Current-year mandi price should only be included
# in a prediction if it is actually available by the
# chosen forecast date. We will decide the final
# prediction-time feature set later.

# --------------------------------------------------
# 7. Farm and calendar features
# --------------------------------------------------

df["log_area_hectares"] = np.log1p(
    df["area_hectares"].clip(lower=0)
)

df["years_since_2015"] = df["year"] - 2015

# --------------------------------------------------
# 8. Historical yield features
# --------------------------------------------------
# The synthetic data has no farm/plot identifier.
# Therefore, historical features are district-crop
# averages, not farm-specific yield histories.
#
# For each district and crop:
# 1. Calculate the average yield for each year.
# 2. Shift by one year.
# 3. Calculate rolling averages using only prior years.
#
# The current year's target is never used to construct
# the historical features for records in that same year.

yearly_yield = (
    df.groupby(["district", "crop", "year"])["yield_q_per_ha"]
    .mean()
    .reset_index()
)

historical_parts = []

all_years = sorted(df["year"].dropna().unique())

for (district, crop), group in yearly_yield.groupby(
    ["district", "crop"]
):

    group = group.set_index("year").sort_index()

    # Include all project years so a missing year
    # does not accidentally become the previous year.
    group = group.reindex(all_years)

    # Shift before rolling to ensure only past yields
    # contribute to each year's historical features.
    previous_yield = group["yield_q_per_ha"].shift(1)

    historical = pd.DataFrame({
        "district": district,
        "crop": crop,
        "year": all_years,
        "previous_year_yield_q_per_ha": previous_yield.values,
        "rolling_3_year_yield_q_per_ha": (
            previous_yield.rolling(
                window=3,
                min_periods=1
            ).mean().values
        )
    })

    historical_parts.append(historical)

historical_features = pd.concat(
    historical_parts,
    ignore_index=True
)

df = df.merge(
    historical_features,
    on=["district", "crop", "year"],
    how="left",
    validate="many_to_one"
)

# --------------------------------------------------
# 9. Add historical rainfall feature
# --------------------------------------------------
# Use the district's previous-year average cumulative
# rainfall. This is based on predictor data, not yield.

yearly_rainfall = (
    df.groupby(["district", "year"])["cum_rainfall_cutoff_mm"]
    .mean()
    .reset_index()
)

rainfall_parts = []

for district, group in yearly_rainfall.groupby("district"):

    group = group.set_index("year").sort_index()
    group = group.reindex(all_years)

    previous_rainfall = (
        group["cum_rainfall_cutoff_mm"].shift(1)
    )

    rainfall_parts.append(
        pd.DataFrame({
            "district": district,
            "year": all_years,
            "previous_year_rainfall_mm": previous_rainfall.values
        })
    )

historical_rainfall = pd.concat(
    rainfall_parts,
    ignore_index=True
)

df = df.merge(
    historical_rainfall,
    on=["district", "year"],
    how="left",
    validate="many_to_one"
)

# --------------------------------------------------
# 10. Check engineered features
# --------------------------------------------------

engineered_features = [
    "rainfall_adequacy_ratio",
    "rainfall_difference_mm",
    "rainfall_per_dry_spell_day",
    "weather_stress_index",
    "total_npk",
    "nitrogen_phosphorus_ratio",
    "potassium_nitrogen_ratio",
    "soil_ph_distance_from_6_5",
    "price_to_msp_ratio",
    "price_premium_over_msp",
    "log_area_hectares",
    "years_since_2015",
    "previous_year_yield_q_per_ha",
    "rolling_3_year_yield_q_per_ha",
    "previous_year_rainfall_mm"
]

print("\nNEW FEATURES CREATED:")
for feature in engineered_features:
    print("-", feature)

print("\nTotal new engineered features:", len(engineered_features))

# --------------------------------------------------
# 11. Check missing values in engineered features
# --------------------------------------------------

print("\nMISSING VALUES IN ENGINEERED FEATURES:")

print(
    df[engineered_features]
    .isnull()
    .sum()
    .sort_values(ascending=False)
)

# Missing historical yield in early years is expected:
# there is no prior-year history for those records.
# Missing rainfall-derived features will be imputed
# later using training-only statistics.

# --------------------------------------------------
# 12. Save engineered dataset
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 65)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 65)

print("Final dataset shape:", df.shape)
print("Output file:", OUTPUT_PATH)

print("\nSample engineered records:")

print(
    df[
        [
            "district",
            "crop",
            "year",
            "yield_q_per_ha",
            "rainfall_adequacy_ratio",
            "total_npk",
            "price_to_msp_ratio",
            "previous_year_yield_q_per_ha",
            "rolling_3_year_yield_q_per_ha"
        ]
    ].head(10).to_string(index=False)
)

print("\nEngineered dataset saved successfully.")
