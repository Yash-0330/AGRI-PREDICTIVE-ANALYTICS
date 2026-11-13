from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "vidarbha_agri_350k.csv"
PLOT_DIR = ROOT / "reports" / "eda_plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run the dataset generation script first."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("AGRI PREDICTIVE ANALYTICS - EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print(f"\nDataset shape: {df.shape}")
print(f"Total records: {len(df):,}")

# --------------------------------------------------
# 3. District-wise record distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

district_order = df["district"].value_counts().index

sns.countplot(
    data=df,
    x="district",
    order=district_order
)

plt.title("Number of Agricultural Records by District")
plt.xlabel("District")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(PLOT_DIR / "01_district_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 4. Crop-wise record distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="crop",
    order=df["crop"].value_counts().index
)

plt.title("Number of Records by Crop")
plt.xlabel("Crop")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(PLOT_DIR / "02_crop_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 5. Year-wise record distribution
# --------------------------------------------------

year_counts = df["year"].value_counts().sort_index()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=year_counts.index,
    y=year_counts.values
)

plt.title("Agricultural Records by Year")
plt.xlabel("Year")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(PLOT_DIR / "03_year_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 6. Rainfall distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="lpa_rainfall_mm",
    bins=40,
    kde=True
)

plt.title("Distribution of Long-Term Average Rainfall")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(PLOT_DIR / "04_rainfall_distribution.png", dpi=300)
plt.show()

# --------------------------------------------------
# 7. District-wise rainfall comparison
# --------------------------------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="district",
    y="lpa_rainfall_mm"
)

plt.title("Long-Term Average Rainfall by District")
plt.xlabel("District")
plt.ylabel("Rainfall (mm)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(PLOT_DIR / "05_district_rainfall.png", dpi=300)
plt.show()

# --------------------------------------------------
# 8. Soil parameter distributions
# --------------------------------------------------

soil_columns = [
    "soil_nitrogen",
    "soil_phosphorus",
    "soil_potassium",
    "soil_organic_carbon",
    "soil_ph"
]

for column in soil_columns:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x=column,
        bins=35,
        kde=True
    )

    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.tight_layout()

    filename = f"soil_{column}.png"
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.show()

# --------------------------------------------------
# 9. Soil characteristics by district
# --------------------------------------------------

district_soil = (
    df.groupby("district")[soil_columns]
    .mean()
    .round(2)
)

print("\nAVERAGE SOIL CHARACTERISTICS BY DISTRICT")
print(district_soil)

plt.figure(figsize=(12, 7))

sns.heatmap(
    district_soil,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu"
)

plt.title("Average Soil Characteristics by District")
plt.tight_layout()

plt.savefig(PLOT_DIR / "soil_district_heatmap.png", dpi=300)
plt.show()

# --------------------------------------------------
# 10. Rainfall and crop distribution
# --------------------------------------------------

rainfall_crop = (
    df.groupby("crop")["cum_rainfall_cutoff_mm"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=rainfall_crop.index,
    y=rainfall_crop.values
)

plt.title("Average Cumulative Rainfall by Crop")
plt.xlabel("Crop")
plt.ylabel("Cumulative Rainfall (mm)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(PLOT_DIR / "06_crop_rainfall.png", dpi=300)
plt.show()

# --------------------------------------------------
# 11. Raw yield distribution
# --------------------------------------------------
# WARNING:
# yield_raw contains mixed units:
# q/ha, kg/ha and tonnes/ha.
# This chart is for data-quality inspection only.

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="yield_raw",
    bins=50,
    kde=True
)

plt.title("Raw Yield Distribution (Mixed Units - Diagnostic Only)")
plt.xlabel("Recorded Raw Yield Value")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(PLOT_DIR / "07_raw_yield_diagnostic.png", dpi=300)
plt.show()

# --------------------------------------------------
# 12. Raw mandi price distribution
# --------------------------------------------------
# WARNING:
# mandi_price_raw contains mixed price units.
# Do not interpret this as a standardized price.

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="mandi_price_raw",
    bins=50,
    kde=True
)

plt.title("Raw Mandi Price Distribution (Mixed Units)")
plt.xlabel("Recorded Raw Mandi Price")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(PLOT_DIR / "08_raw_mandi_price_diagnostic.png", dpi=300)
plt.show()

# --------------------------------------------------
# 13. Correlation analysis of selected features
# --------------------------------------------------
# Exclude raw yield and mandi price because their
# units are not standardized yet.
# Also exclude record_id and post-harvest leakage.

correlation_columns = [
    "year",
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
    "msp_quintal"
]

correlation_data = df[correlation_columns].corr(numeric_only=True)

plt.figure(figsize=(13, 10))

sns.heatmap(
    correlation_data,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Selected Numerical Features")
plt.tight_layout()

plt.savefig(PLOT_DIR / "09_feature_correlation.png", dpi=300)
plt.show()

# --------------------------------------------------
# 14. Missing-value visualization
# --------------------------------------------------

missing_percent = (
    df.isnull().mean() * 100
).sort_values(ascending=False)

missing_percent = missing_percent[missing_percent > 0]

print("\nMISSING VALUE PERCENTAGES")
print(missing_percent.round(2))

if not missing_percent.empty:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=missing_percent.values,
        y=missing_percent.index
    )

    plt.title("Missing Values by Column")
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Column")
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "10_missing_values.png", dpi=300)
    plt.show()

# --------------------------------------------------
# 15. Important EDA findings
# --------------------------------------------------

print("\n" + "=" * 65)
print("EDA SUMMARY")
print("=" * 65)

print("\n1. Districts:")
print(df["district"].nunique())

print("\n2. Crops:")
print(df["crop"].nunique())

print("\n3. Years covered:")
print(df["year"].min(), "to", df["year"].max())

print("\n4. Missing rainfall records:")
print(df["lpa_rainfall_mm"].isnull().sum())

print("\n5. Duplicate rows:")
print(df.duplicated().sum())

print("\n6. Duplicate record IDs:")
print(df["record_id"].duplicated().sum())

print("\n7. Raw yield range:")
print(df["yield_raw"].min(), "to", df["yield_raw"].max())

print("\n8. Raw mandi price range:")
print(df["mandi_price_raw"].min(), "to", df["mandi_price_raw"].max())

print("\n9. Leakage column:")
print("post_harvest_mandi_arrivals_tonnes")
print("Must not be used as a prediction feature.")

print("\n" + "=" * 65)
print("EDA COMPLETED SUCCESSFULLY!")
print(f"All charts saved in: {PLOT_DIR}")
print("=" * 65)
