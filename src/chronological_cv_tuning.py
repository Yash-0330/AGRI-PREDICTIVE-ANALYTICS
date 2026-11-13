
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "model_ready"
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD TRAINING DATA
# --------------------------------------------------

print("=" * 70)
print("CHRONOLOGICAL CROSS-VALIDATION")
print("=" * 70)

X = load_npz(DATA_DIR / "X_train.npz")
y = np.load(DATA_DIR / "y_train.npy")

train_df = pd.read_csv(SPLIT_DIR / "train.csv")

years = train_df["year"].to_numpy()

if X.shape[0] != len(y) or X.shape[0] != len(years):
    raise ValueError(
        "Feature, target and year row counts do not match. "
        "Check that preprocessing used the same training CSV."
    )

print("Training records:", X.shape[0])
print("Feature count:", X.shape[1])
print("Available years:", sorted(np.unique(years)))


# --------------------------------------------------
# 3. DEFINE CHRONOLOGICAL FOLDS
# --------------------------------------------------

folds = [
    (2019, 2020),
    (2020, 2021),
    (2021, 2022)
]

# Each tuple means:
# Train on years <= first value.
# Validate on the second value.


# --------------------------------------------------
# 4. DEFINE MODEL CONFIGURATIONS
# --------------------------------------------------

model_configs = {
    "Ridge": {
        "model": "ridge"
    },

    "LightGBM_Default": {
        "model": "lightgbm",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "reg_lambda": 1.0
    },

    "LightGBM_Tuned_A": {
        "model": "lightgbm",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 20,
        "max_depth": 8,
        "reg_lambda": 2.0
    },

    "LightGBM_Tuned_B": {
        "model": "lightgbm",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 40,
        "max_depth": 10,
        "reg_lambda": 3.0
    }
}


# --------------------------------------------------
# 5. MODEL CREATION FUNCTION
# --------------------------------------------------

def create_model(config):

    if config["model"] == "ridge":
        return Ridge(alpha=1.0)

    return LGBMRegressor(
        objective="regression",
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        num_leaves=config["num_leaves"],
        max_depth=config["max_depth"],
        reg_lambda=config["reg_lambda"],
        max_bin=127,
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )


# --------------------------------------------------
# 6. CHRONOLOGICAL CROSS-VALIDATION
# --------------------------------------------------

cv_results = []

for fold_number, (last_train_year, validation_year) in enumerate(folds, 1):

    train_idx = np.where(years <= last_train_year)[0]
    val_idx = np.where(years == validation_year)[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"No records found for fold {fold_number}. "
            "Check the available years."
        )

    X_fold_train = X[train_idx]
    X_fold_val = X[val_idx]

    y_fold_train = y[train_idx]
    y_fold_val = y[val_idx]

    print("\n" + "-" * 70)
    print(f"FOLD {fold_number}")
    print(f"Training through: {last_train_year}")
    print(f"Validation year: {validation_year}")
    print("Training records:", len(train_idx))
    print("Validation records:", len(val_idx))

    for model_name, config in model_configs.items():

        model = create_model(config)

        start = time.perf_counter()

        model.fit(X_fold_train, y_fold_train)

        training_time = time.perf_counter() - start

        predictions = model.predict(X_fold_val)

        mae = mean_absolute_error(y_fold_val, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_fold_val, predictions)
        )
        r2 = r2_score(y_fold_val, predictions)

        cv_results.append({
            "Fold": fold_number,
            "Model": model_name,
            "Train_Through_Year": last_train_year,
            "Validation_Year": validation_year,
            "Training_Rows": len(train_idx),
            "Validation_Rows": len(val_idx),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Training_Time_Seconds": training_time
        })

        print(
            f"{model_name}: "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R2={r2:.4f}"
        )


# --------------------------------------------------
# 7. SAVE FOLD RESULTS
# --------------------------------------------------

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    REPORT_DIR / "chronological_cv_results.csv",
    index=False
)


# --------------------------------------------------
# 8. CALCULATE AVERAGE PERFORMANCE
# --------------------------------------------------

summary = (
    cv_df.groupby("Model")
    .agg(
        Mean_MAE=("MAE", "mean"),
        Mean_RMSE=("RMSE", "mean"),
        Mean_R2=("R2", "mean"),
        MAE_Std=("MAE", "std")
    )
    .reset_index()
    .sort_values("Mean_MAE")
)

summary.to_csv(
    REPORT_DIR / "chronological_cv_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("CHRONOLOGICAL CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))


# --------------------------------------------------
# 9. SELECT LIGHTGBM CONFIGURATION
# --------------------------------------------------

lightgbm_summary = summary[
    summary["Model"].str.startswith("LightGBM")
]

best_config_name = lightgbm_summary.iloc[0]["Model"]

selected_config = model_configs[best_config_name]

print("\nBest LightGBM configuration:", best_config_name)


# --------------------------------------------------
# 10. TRAIN SELECTED CONFIGURATION ON FULL TRAIN DATA
# --------------------------------------------------

final_model = create_model(selected_config)

print("\nTraining selected configuration on all training years...")

final_model.fit(X, y)

joblib.dump(
    final_model,
    MODEL_DIR / "tuned_lightgbm_model.joblib"
)

selection = {
    "selected_configuration": best_config_name,
    "selection_method": "Lowest mean chronological CV MAE",
    "mean_cv_mae": float(
        lightgbm_summary.iloc[0]["Mean_MAE"]
    ),
    "mean_cv_rmse": float(
        lightgbm_summary.iloc[0]["Mean_RMSE"]
    ),
    "mean_cv_r2": float(
        lightgbm_summary.iloc[0]["Mean_R2"]
    ),
    "hyperparameters": {
        key: value
        for key, value in selected_config.items()
        if key != "model"
    }
}

with open(
    MODEL_DIR / "tuned_lightgbm_selection.json",
    "w"
) as file:
    json.dump(selection, file, indent=4)

print("\nCross-validation and tuning completed.")
print("Results:", REPORT_DIR / "chronological_cv_summary.csv")
print("Tuned model:", MODEL_DIR / "tuned_lightgbm_model.joblib")

print("\nImportant: The 2024 test dataset was not used.")
