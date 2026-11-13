
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "model_ready"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

SHAP_DIR = REPORT_DIR / "shap_analysis"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("MODEL EXPLAINABILITY - SHAP ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# 2. LOAD MODEL AND TEST FEATURES
# --------------------------------------------------

model_path = MODEL_DIR / "tuned_lightgbm_model.joblib"
features_path = DATA_DIR / "X_test.npz"
names_path = DATA_DIR / "feature_names.json"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not features_path.exists():
    raise FileNotFoundError(f"Test features not found: {features_path}")

if not names_path.exists():
    raise FileNotFoundError(f"Feature names not found: {names_path}")

model = joblib.load(model_path)
X_test = load_npz(features_path)

print("Test matrix shape:", X_test.shape)


# --------------------------------------------------
# 3. LOAD FEATURE NAMES
# --------------------------------------------------

with open(names_path, "r") as file:
    feature_data = json.load(file)

if isinstance(feature_data, list):
    feature_names = feature_data

elif isinstance(feature_data, dict):
    feature_names = feature_data.get("feature_names")

    if feature_names is None:
        raise ValueError(
            "Could not find 'feature_names' in feature_names.json. "
            "Inspect the JSON file and update this section."
        )

else:
    raise ValueError("Unexpected feature_names.json format.")

if len(feature_names) != X_test.shape[1]:
    raise ValueError(
        f"Feature name count ({len(feature_names)}) does not match "
        f"matrix columns ({X_test.shape[1]})."
    )

print("Feature names loaded:", len(feature_names))


# --------------------------------------------------
# 4. SELECT A REPRODUCIBLE TEST SAMPLE
# --------------------------------------------------

sample_size = min(1000, X_test.shape[0])

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    X_test.shape[0],
    size=sample_size,
    replace=False
)

X_sample = X_test[sample_indices].toarray()

X_sample_df = pd.DataFrame(
    X_sample,
    columns=feature_names
)

print("SHAP sample size:", sample_size)


# --------------------------------------------------
# 5. CREATE SHAP EXPLAINER
# --------------------------------------------------

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample_df)

# Handle both single-output array and newer SHAP formats.
if isinstance(shap_values, list):
    shap_values = shap_values[0]

if hasattr(shap_values, "values"):
    shap_values = shap_values.values

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

if shap_values.shape != X_sample_df.shape:
    raise ValueError(
        f"Unexpected SHAP output shape: {shap_values.shape}"
    )

print("SHAP calculation completed.")


# --------------------------------------------------
# 6. CALCULATE GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": importance
})

importance_df = importance_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

importance_df.to_csv(
    SHAP_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTOP 15 IMPORTANT FEATURES")
print(importance_df.head(15).to_string(index=False))


# --------------------------------------------------
# 7. SHAP SUMMARY PLOT
# --------------------------------------------------

plt.figure(figsize=(11, 8))

shap.summary_plot(
    shap_values,
    X_sample_df,
    show=False,
    max_display=15
)

plt.title("SHAP Summary - Crop Yield Prediction")
plt.tight_layout()

plt.savefig(
    SHAP_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 8. SHAP BAR PLOT
# --------------------------------------------------

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values,
    X_sample_df,
    plot_type="bar",
    show=False,
    max_display=15
)

plt.title("Global Feature Importance - SHAP")
plt.tight_layout()

plt.savefig(
    SHAP_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 9. SAVE TOP FEATURES
# --------------------------------------------------

top_features = importance_df.head(10)

top_features.to_csv(
    SHAP_DIR / "top_10_features.csv",
    index=False
)


# --------------------------------------------------
# 10. EXPORT SAMPLE EXPLANATIONS
# --------------------------------------------------

explanation_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)

explanation_df.insert(
    0,
    "sample_test_row",
    sample_indices
)

explanation_df.to_csv(
    SHAP_DIR / "sample_shap_values.csv",
    index=False
)


# --------------------------------------------------
# 11. BUSINESS INTERPRETATION TEMPLATE
# --------------------------------------------------

business_notes = []

for _, row in top_features.iterrows():

    business_notes.append({
        "Feature": row["Feature"],
        "Mean_Absolute_SHAP": row["Mean_Absolute_SHAP"],
        "Business_Interpretation": (
            "Investigate this feature's relationship with crop yield "
            "using domain knowledge and the SHAP summary plot."
        )
    })

business_df = pd.DataFrame(business_notes)

business_df.to_csv(
    SHAP_DIR / "business_interpretation_template.csv",
    index=False
)


# --------------------------------------------------
# 12. COMPLETION
# --------------------------------------------------

print("\nSHAP ANALYSIS COMPLETED")
print("Output directory:", SHAP_DIR)

print("\nGenerated files:")
for path in sorted(SHAP_DIR.iterdir()):
    if path.is_file():
        print("-", path.name)
