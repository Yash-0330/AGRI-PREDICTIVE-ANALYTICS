from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, save_npz
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = ROOT / "data" / "processed" / "model_ready"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load split datasets
# --------------------------------------------------

train_path = SPLIT_DIR / "train.csv"
validation_path = SPLIT_DIR / "validation.csv"
test_path = SPLIT_DIR / "test.csv"

for path in [train_path, validation_path, test_path]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file: {path}\n"
            "Run python src/split_dataset.py first."
        )

train = pd.read_csv(train_path)
validation = pd.read_csv(validation_path)
test = pd.read_csv(test_path)

print("=" * 70)
print("MACHINE LEARNING PREPROCESSING PIPELINE")
print("=" * 70)

print("\nLoaded dataset sizes:")
print("Training:", train.shape)
print("Validation:", validation.shape)
print("Testing:", test.shape)

# --------------------------------------------------
# 3. Define target
# --------------------------------------------------

TARGET = "yield_q_per_ha"

# --------------------------------------------------
# 4. Explicitly select model features
# --------------------------------------------------
# Do not use all columns automatically.
#
# Excluded intentionally:
# - yield_raw and yield_unit_detected
# - target-derived invalid/outlier flags
# - current-year mandi prices and price ratios
# - record_id
# - post-harvest information
#
# Current-year mandi price is excluded because it may
# not be available at the intended pre-harvest forecast
# date. MSP is retained as a known policy feature.

numeric_features = [
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
    "msp_quintal",
    "rainfall_was_missing",
    "rainfall_adequacy_ratio",
    "rainfall_difference_mm",
    "rainfall_per_dry_spell_day",
    "weather_stress_index",
    "total_npk",
    "nitrogen_phosphorus_ratio",
    "potassium_nitrogen_ratio",
    "soil_ph_distance_from_6_5",
    "log_area_hectares",
    "years_since_2015",
    "previous_year_yield_q_per_ha",
    "rolling_3_year_yield_q_per_ha",
    "previous_year_rainfall_mm"
]

categorical_features = [
    "district",
    "taluka",
    "crop",
    "season"
]

feature_columns = numeric_features + categorical_features

# --------------------------------------------------
# 5. Validate feature availability
# --------------------------------------------------

for dataset_name, dataset in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    missing_features = [
        column for column in feature_columns
        if column not in dataset.columns
    ]

    if missing_features:
        raise ValueError(
            f"{dataset_name} dataset is missing features: "
            f"{missing_features}"
        )

    if TARGET not in dataset.columns:
        raise ValueError(
            f"{TARGET} is missing from {dataset_name} dataset."
        )

# --------------------------------------------------
# 6. Separate X and y
# --------------------------------------------------

X_train = train[feature_columns].copy()
y_train = train[TARGET].copy()

X_validation = validation[feature_columns].copy()
y_validation = validation[TARGET].copy()

X_test = test[feature_columns].copy()
y_test = test[TARGET].copy()

# --------------------------------------------------
# 7. Build numerical preprocessing pipeline
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

# --------------------------------------------------
# 8. Build categorical preprocessing pipeline
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)

# --------------------------------------------------
# 9. Combine preprocessing pipelines
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)

# --------------------------------------------------
# 10. Fit ONLY on training data
# --------------------------------------------------

print("\nFitting preprocessing pipeline on training data...")

X_train_processed = preprocessor.fit_transform(X_train)

# Validation and test are transformed only.
X_validation_processed = preprocessor.transform(X_validation)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed.")

# --------------------------------------------------
# 11. Convert to CSR sparse matrices
# --------------------------------------------------

X_train_processed = csr_matrix(X_train_processed)

X_validation_processed = csr_matrix(
    X_validation_processed
)

X_test_processed = csr_matrix(
    X_test_processed
)

print("\nConverted all datasets into CSR sparse matrices.")

# --------------------------------------------------
# 12. Save transformed feature matrices
# --------------------------------------------------

save_npz(
    OUTPUT_DIR / "X_train.npz",
    X_train_processed
)

save_npz(
    OUTPUT_DIR / "X_validation.npz",
    X_validation_processed
)

save_npz(
    OUTPUT_DIR / "X_test.npz",
    X_test_processed
)

# Save target arrays.
np.save(OUTPUT_DIR / "y_train.npy", y_train.to_numpy())
np.save(OUTPUT_DIR / "y_validation.npy", y_validation.to_numpy())
np.save(OUTPUT_DIR / "y_test.npy", y_test.to_numpy())

# --------------------------------------------------
# 13. Save preprocessing model
# --------------------------------------------------

preprocessor_path = MODEL_DIR / "preprocessor.joblib"

joblib.dump(preprocessor, preprocessor_path)

# --------------------------------------------------
# 14. Save feature names
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out().tolist()

with open(
    OUTPUT_DIR / "feature_names.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(feature_names, file, indent=4)

# --------------------------------------------------
# 15. Save metadata
# --------------------------------------------------

metadata = {
    "target": TARGET,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "all_input_features": feature_columns,
    "encoded_feature_count": len(feature_names),
    "training_rows": len(X_train),
    "validation_rows": len(X_validation),
    "testing_rows": len(X_test),
    "preprocessing_fit_dataset": "training_only"
}

with open(
    OUTPUT_DIR / "preprocessing_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

# --------------------------------------------------
# 16. Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING RESULTS")
print("=" * 70)

print("\nTraining matrix:", X_train_processed.shape)
print("Validation matrix:", X_validation_processed.shape)
print("Testing matrix:", X_test_processed.shape)

print("\nEncoded feature count:", len(feature_names))

print("\nTarget array sizes:")
print("Training:", y_train.shape)
print("Validation:", y_validation.shape)
print("Testing:", y_test.shape)

print("\nSaved preprocessing model:")
print(preprocessor_path)

print("\nSaved model-ready files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\n" + "=" * 70)
print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)