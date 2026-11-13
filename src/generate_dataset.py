
import numpy as np
import pandas as pd
from pathlib import Path


def generate_vidarbha_agri_data(n_rows=350000, seed=42):

    rng = np.random.default_rng(seed)

    # -----------------------------------
    # 1. DISTRICT AND CROP INFORMATION
    # -----------------------------------

    districts = [
        "Nagpur", "Wardha", "Amravati",
        "Yavatmal", "Akola", "Chandrapur"
    ]

    district_rain_base = {
        "Chandrapur": 1150,
        "Nagpur": 1020,
        "Wardha": 950,
        "Yavatmal": 880,
        "Amravati": 820,
        "Akola": 760
    }

    crops = [
        "Soybean", "Cotton", "Tur", "Wheat", "Gram"
    ]

    crop_base_yield_qha = {
        "Soybean": 13.5,
        "Cotton": 11.0,
        "Tur": 10.5,
        "Wheat": 18.0,
        "Gram": 12.0
    }

    crop_msp_quintal = {
        "Soybean": 4600,
        "Cotton": 6620,
        "Tur": 7000,
        "Wheat": 2275,
        "Gram": 5440
    }

    # -----------------------------------
    # 2. GENERATE YEAR AND CATEGORIES
    # -----------------------------------

    years = rng.choice(
        np.arange(2015, 2025),
        size=n_rows,
        p=[0.09, 0.09, 0.10, 0.10, 0.10,
           0.10, 0.10, 0.10, 0.11, 0.11]
    )

    dist_col = rng.choice(districts, size=n_rows)

    taluka_col = [
        f"{d}_Taluka_{rng.integers(1, 9)}"
        for d in dist_col
    ]

    crop_col = rng.choice(
        crops,
        size=n_rows,
        p=[0.35, 0.30, 0.15, 0.10, 0.10]
    )

    season_map = {
        "Soybean": "Kharif",
        "Cotton": "Kharif",
        "Tur": "Kharif",
        "Wheat": "Rabi",
        "Gram": "Rabi"
    }

    season_col = [season_map[c] for c in crop_col]

    # -----------------------------------
    # 3. SOIL HEALTH PARAMETERS
    # -----------------------------------

    soil_n = rng.normal(
        210, 35, n_rows
    ).clip(100, 350)

    soil_p = rng.normal(
        18, 5, n_rows
    ).clip(5, 40)

    soil_k = rng.normal(
        320, 60, n_rows
    ).clip(120, 550)

    soil_oc = rng.normal(
        0.55, 0.15, n_rows
    ).clip(0.2, 1.2)

    soil_ph = rng.normal(
        7.2, 0.55, n_rows
    ).clip(5.5, 8.8)

    # -----------------------------------
    # 4. RAINFALL INFORMATION
    # -----------------------------------

    lpa_rain = np.array([
        district_rain_base[d] for d in dist_col
    ])

    rain_anomaly = rng.normal(
        1.0, 0.22, n_rows
    ).clip(0.45, 1.65)

    cum_rain_mm = (
        lpa_rain * 0.65 * rain_anomaly
    )

    dry_spell_days = np.where(
        rain_anomaly < 0.85,
        rng.integers(12, 26, n_rows),
        rng.integers(3, 14, n_rows)
    )

    heavy_rain_days = np.where(
        rain_anomaly > 1.25,
        rng.integers(3, 9, n_rows),
        rng.integers(0, 3, n_rows)
    )

    # -----------------------------------
    # 5. MANDI MARKET PRICES
    # -----------------------------------

    msp = np.array([
        crop_msp_quintal[c] for c in crop_col
    ])

    mandi_price = (
        msp * rng.normal(1.02, 0.14, n_rows)
    )

    # -----------------------------------
    # 6. AGRICULTURAL YIELD GENERATION
    # -----------------------------------

    base_yield = np.array([
        crop_base_yield_qha[c] for c in crop_col
    ])

    rain_effect = (
        -6.5 * (rain_anomaly - 1.05) ** 2 + 1.2
    )

    dry_penalty = (
        -0.18 * np.maximum(0, dry_spell_days - 10)
    )

    soil_effect = (
        2.1 * (soil_oc - 0.55)
        + 0.005 * (soil_n - 210)
    )

    price_incentive = (
        1.4 * ((mandi_price / msp) - 1.0)
    )

    true_yield_qha = (
        base_yield
        + rain_effect
        + dry_penalty
        + soil_effect
        + price_incentive
        + rng.normal(0, 1.35, n_rows)
    ).clip(2.0, 32.0)

    # -----------------------------------
    # 7. CULTIVATED AREA
    # -----------------------------------

    area_hectares = rng.lognormal(
        mean=1.5, sigma=0.8, size=n_rows
    ).clip(0.5, 100)

    # -----------------------------------
    # 8. CREATE DATAFRAME
    # -----------------------------------

    df = pd.DataFrame({

        "record_id": np.arange(1, n_rows + 1),

        "year": years,

        "district": dist_col,

        "taluka": taluka_col,

        "crop": crop_col,

        "season": season_col,

        "area_hectares": area_hectares,

        "lpa_rainfall_mm": lpa_rain,

        "cum_rainfall_cutoff_mm": cum_rain_mm,

        "dry_spell_max_days": dry_spell_days,

        "heavy_rain_days": heavy_rain_days,

        "soil_nitrogen": soil_n,

        "soil_phosphorus": soil_p,

        "soil_potassium": soil_k,

        "soil_organic_carbon": soil_oc,

        "soil_ph": soil_ph,

        "mandi_price_raw": mandi_price,

        "msp_quintal": msp,

        "yield_raw": true_yield_qha,

        "post_harvest_mandi_arrivals_tonnes":
            true_yield_qha * rng.uniform(80, 150, n_rows)

    })

    # -----------------------------------
    # 9. INJECT PRICE UNIT ERRORS
    # -----------------------------------

    kg_mask = rng.random(n_rows) < 0.18

    df.loc[kg_mask, "mandi_price_raw"] /= 100.0

    # -----------------------------------
    # 10. INJECT YIELD UNIT ERRORS
    # -----------------------------------

    kg_yield_mask = rng.random(n_rows) < 0.12

    tonne_yield_mask = (
        (~kg_yield_mask)
        & (rng.random(n_rows) < 0.09)
    )

    df.loc[kg_yield_mask, "yield_raw"] *= 100.0

    df.loc[tonne_yield_mask, "yield_raw"] /= 10.0

    # -----------------------------------
    # 11. INJECT MISSING RAINFALL
    # -----------------------------------

    rain_na_mask = rng.random(n_rows) < 0.14

    df.loc[
        rain_na_mask, "cum_rainfall_cutoff_mm"
    ] = np.nan

    # -----------------------------------
    # 12. INJECT DUPLICATE RECORDS
    # -----------------------------------

    duplicate_rows = df.sample(
        frac=0.015,
        random_state=seed
    )

    df = pd.concat(
        [df, duplicate_rows],
        ignore_index=True
    )

    return df


# -----------------------------------
# 13. RUN DATASET GENERATOR
# -----------------------------------

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    output_folder = project_root / "data" / "raw"

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_folder / "vidarbha_agri_350k.csv"
    )

    print("Generating Vidarbha agricultural dataset...")

    df = generate_vidarbha_agri_data(
        n_rows=350000,
        seed=42
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\nDATASET GENERATION COMPLETED")

    print("Dataset shape:", df.shape)

    print("Number of records:", len(df))

    print("Number of columns:", len(df.columns))

    print("Missing rainfall values:",
          df["cum_rainfall_cutoff_mm"].isna().sum())

    print("Unique districts:",
          df["district"].nunique())

    print("Unique crops:",
          df["crop"].nunique())

    print("Dataset saved at:", output_file)
