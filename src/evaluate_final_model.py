
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "model_ready"
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PLOT_DIR = REPORT_DIR / "final_evaluation_plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("FINAL MODEL EVALUATION - 2024 TEST DATA")
print("=" * 70)


# --------------------------------------------------
# 1. LOAD MODEL AND TEST DATA
# --------------------------------------------------

model_path = MODEL_DIR / "tuned_lightgbm_model.joblib"

if not model_path.exists():
    raise FileNotFoundError(
        "Tuned model not found. Complete Step 9 first."
    )

model = joblib.load(model_path)

X_test = load_npz(DATA_DIR / "X_test.npz")
y_test = np.load(DATA_DIR / "y_test.npy")

test_df = pd.read_csv(SPLIT_DIR / "test.csv")

if X_test.shape[0] != len(y_test):
    raise ValueError("Test feature and target counts do not match.")

if X_test.shape[0] != len(test_df):
    raise ValueError("Test CSV and processed feature counts do not match.")

print("Test records:", X_test.shape[0])
print("Feature count:", X_test.shape[1])


# --------------------------------------------------
# 2. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

predictions = model.predict(X_test)

# Crop yield cannot be negative.
# Preserve raw predictions separately for diagnostics.
raw_predictions = predictions.copy()
predictions = np.maximum(predictions, 0)


# --------------------------------------------------
# 3. CALCULATE PERFORMANCE METRICS
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

metrics = {
    "Test_Year": 2024,
    "Test_Records": int(len(y_test)),
    "MAE_q_per_ha": float(mae),
    "RMSE_q_per_ha": float(rmse),
    "R2": float(r2),
    "Mean_Actual_Yield": float(np.mean(y_test)),
    "Mean_Predicted_Yield": float(np.mean(predictions))
}

print("\nFINAL TEST PERFORMANCE")
print("-" * 50)

for key, value in metrics.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

with open(REPORT_DIR / "final_test_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# --------------------------------------------------
# 4. SAVE PREDICTIONS
# --------------------------------------------------

prediction_df = test_df.copy()

prediction_df["actual_yield_q_per_ha"] = y_test
prediction_df["raw_predicted_yield_q_per_ha"] = raw_predictions
prediction_df["predicted_yield_q_per_ha"] = predictions

prediction_df["prediction_error"] = (
    prediction_df["actual_yield_q_per_ha"]
    - prediction_df["predicted_yield_q_per_ha"]
)

prediction_df["absolute_error"] = (
    prediction_df["prediction_error"].abs()
)

prediction_df["prediction_status"] = np.where(
    prediction_df["prediction_error"] > 0,
    "Underprediction",
    np.where(
        prediction_df["prediction_error"] < 0,
        "Overprediction",
        "Exact"
    )
)

prediction_df.to_csv(
    REPORT_DIR / "test_predictions_2024.csv",
    index=False
)


# --------------------------------------------------
# 5. DISTRICT-LEVEL PERFORMANCE
# --------------------------------------------------

district_metrics = (
    prediction_df.groupby("district")
    .apply(
        lambda group: pd.Series({
            "Records": len(group),
            "MAE": mean_absolute_error(
                group["actual_yield_q_per_ha"],
                group["predicted_yield_q_per_ha"]
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    group["actual_yield_q_per_ha"],
                    group["predicted_yield_q_per_ha"]
                )
            ),
            "Mean_Actual_Yield": group[
                "actual_yield_q_per_ha"
            ].mean(),
            "Mean_Predicted_Yield": group[
                "predicted_yield_q_per_ha"
            ].mean()
        })
    )
    .reset_index()
)

district_metrics.to_csv(
    REPORT_DIR / "district_test_performance.csv",
    index=False
)


# --------------------------------------------------
# 6. CROP-LEVEL PERFORMANCE
# --------------------------------------------------

crop_metrics = (
    prediction_df.groupby("crop")
    .apply(
        lambda group: pd.Series({
            "Records": len(group),
            "MAE": mean_absolute_error(
                group["actual_yield_q_per_ha"],
                group["predicted_yield_q_per_ha"]
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    group["actual_yield_q_per_ha"],
                    group["predicted_yield_q_per_ha"]
                )
            ),
            "Mean_Actual_Yield": group[
                "actual_yield_q_per_ha"
            ].mean(),
            "Mean_Predicted_Yield": group[
                "predicted_yield_q_per_ha"
            ].mean()
        })
    )
    .reset_index()
)

crop_metrics.to_csv(
    REPORT_DIR / "crop_test_performance.csv",
    index=False
)


# --------------------------------------------------
# 7. ACTUAL VS PREDICTED PLOT
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.25
)

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Yield (q/ha)")
plt.ylabel("Predicted Yield (q/ha)")
plt.title("Actual vs Predicted Crop Yield - 2024 Test Data")
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 8. RESIDUAL DISTRIBUTION
# --------------------------------------------------

residuals = y_test - predictions

plt.figure(figsize=(8, 6))

plt.hist(
    residuals,
    bins=50,
    edgecolor="black"
)

plt.xlabel("Prediction Error (q/ha)")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "residual_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 9. DISTRICT MAE CHART
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    district_metrics["district"],
    district_metrics["MAE"]
)

plt.xlabel("District")
plt.ylabel("MAE (q/ha)")
plt.title("Prediction Error by District")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "district_mae.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 10. CROP MAE CHART
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    crop_metrics["crop"],
    crop_metrics["MAE"]
)

plt.xlabel("Crop")
plt.ylabel("MAE (q/ha)")
plt.title("Prediction Error by Crop")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "crop_mae.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 11. DISPLAY GROUP METRICS
# --------------------------------------------------

print("\nDISTRICT-LEVEL PERFORMANCE")
print(district_metrics.to_string(index=False))

print("\nCROP-LEVEL PERFORMANCE")
print(crop_metrics.to_string(index=False))


# --------------------------------------------------
# 12. IDENTIFY LARGEST PREDICTION ERRORS
# --------------------------------------------------

worst_predictions = prediction_df.nlargest(
    10,
    "absolute_error"
)

worst_predictions.to_csv(
    REPORT_DIR / "largest_prediction_errors.csv",
    index=False
)

print("\n10 LARGEST PREDICTION ERRORS")
print(
    worst_predictions[
        [
            "district",
            "crop",
            "actual_yield_q_per_ha",
            "predicted_yield_q_per_ha",
            "absolute_error"
        ]
    ].to_string(index=False)
)

print("\nEvaluation completed successfully.")
print("Reports saved in:", REPORT_DIR)
print("Plots saved in:", PLOT_DIR)
