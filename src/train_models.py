
from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lightgbm import LGBMRegressor


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "model_ready"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD PREPROCESSED DATA
# --------------------------------------------------

print("=" * 70)
print("AGRI CROP YIELD - MODEL TRAINING")
print("=" * 70)

X_train = load_npz(DATA_DIR / "X_train.npz")
X_val = load_npz(DATA_DIR / "X_validation.npz")

y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_validation.npy")

print("\nDataset loaded successfully.")
print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Training target:", y_train.shape)
print("Validation target:", y_val.shape)


# --------------------------------------------------
# 3. DEFINE MODELS
# --------------------------------------------------

models = {
    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=10,
        max_features=0.8,
        n_jobs=-1,
        random_state=42
    ),

    "LightGBM": LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        max_bin=127,
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )
}


# --------------------------------------------------
# 4. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(name, actual, predicted, training_seconds):

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Training_Time_Seconds": training_seconds
    }


results = []


# --------------------------------------------------
# 5. BASELINE MODEL
# --------------------------------------------------

print("\n" + "=" * 70)
print("MODEL 1: BASELINE")
print("=" * 70)

baseline = DummyRegressor(strategy="mean")

start = time.perf_counter()
baseline.fit(X_train, y_train)
baseline_time = time.perf_counter() - start

baseline_predictions = baseline.predict(X_val)

results.append(
    evaluate_model(
        "Baseline Mean",
        y_val,
        baseline_predictions,
        baseline_time
    )
)

joblib.dump(baseline, MODEL_DIR / "baseline_model.joblib")

print("Baseline evaluation completed.")


# --------------------------------------------------
# 6. TRAIN OTHER MODELS
# --------------------------------------------------

for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    print("Training completed.")
    print("Generating validation predictions...")

    predictions = model.predict(X_val)

    model_results = evaluate_model(
        name,
        y_val,
        predictions,
        training_time
    )

    results.append(model_results)

    # Save trained model
    filename = name.lower().replace(" ", "_") + "_model.joblib"
    joblib.dump(model, MODEL_DIR / filename)

    print(f"{name} evaluation completed.")
    print(f"Validation MAE: {model_results['MAE']:.4f}")
    print(f"Validation RMSE: {model_results['RMSE']:.4f}")
    print(f"Validation R2: {model_results['R2']:.4f}")
    print(f"Training time: {training_time:.2f} seconds")


# --------------------------------------------------
# 7. COMPARE ALL MODELS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))


# --------------------------------------------------
# 8. SELECT BEST MODEL USING VALIDATION MAE
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

print("\nBest model based on validation MAE:")
print(best_model_name)

# Store the selected model name.
# The final 2024 test evaluation will happen in a later step.

with open(MODEL_DIR / "best_model_selection.json", "w") as file:
    json.dump(
        {
            "selected_model": best_model_name,
            "selection_metric": "Validation MAE",
            "validation_mae": float(results_df.iloc[0]["MAE"]),
            "validation_rmse": float(results_df.iloc[0]["RMSE"]),
            "validation_r2": float(results_df.iloc[0]["R2"])
        },
        file,
        indent=4
    )

print("\nModel comparison completed successfully.")
print("Results saved in:", REPORT_DIR / "model_comparison.csv")
print("Trained models saved in:", MODEL_DIR)
print("\nNote: The 2024 test dataset has NOT been used.")
