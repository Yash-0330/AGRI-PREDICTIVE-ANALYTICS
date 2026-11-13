
# MODEL CARD
## Agri Predictive Analytics Model
### District-Level Crop Yield Prediction and Business Insights

---

## 1. Model Overview

**Project Name:** Agri Predictive Analytics Model

**Model Type:** Supervised Machine Learning - Regression

**Primary Objective:** Predict agricultural crop yield at the district level in Vidarbha, Maharashtra.

**Prediction Target:** Crop yield in quintals per hectare (q/ha).

**Geographical Coverage:**
- Nagpur
- Wardha
- Amravati
- Yavatmal
- Akola
- Chandrapur

**Dataset Period:** 2015–2024

**Dataset Type:** Synthetic agricultural dataset.

---

## 2. Intended Use

The model is designed as a prototype decision-support tool for exploring agricultural yield prediction and business analytics.

Potential applications include:

- District-level crop yield estimation.
- Agricultural procurement planning.
- Crop performance comparison.
- Rainfall and soil condition analysis.
- Agricultural risk assessment.
- Identifying patterns in historical crop yield data.

The model is not intended to replace agricultural experts or official government crop-yield estimates.

---

## 3. Dataset Description

The project uses a synthetically generated dataset containing approximately 350,000 agricultural records before duplicate removal.

The dataset includes:

- District and taluka.
- Crop and agricultural season.
- Cultivated land area.
- Rainfall measurements.
- Soil nitrogen, phosphorus and potassium.
- Soil organic carbon.
- Soil pH.
- Minimum Support Price (MSP).
- Historical yield information.

Data quality issues intentionally introduced include:

- Missing rainfall values.
- Duplicate records.
- Mixed measurement units.
- Unusual observations.
- A post-harvest variable that requires leakage auditing.

**Important:** This dataset is synthetic and does not represent verified agricultural observations.

---

## 4. Data Preparation

The preprocessing workflow includes:

1. Duplicate record removal.
2. Yield unit standardization.
3. Mandi price unit standardization.
4. Missing rainfall identification.
5. Outlier identification.
6. Feature engineering.
7. Historical feature generation.
8. Categorical encoding.
9. Numerical missing-value imputation.
10. Chronological train-validation-test splitting.

The data split is:

| Dataset | Period |
|---|---|
| Training | 2015–2022 |
| Validation | 2023 |
| Testing | 2024 |

The preprocessing pipeline is fitted on training data and applied to validation and test data.

---

## 5. Feature Engineering

Engineered features include:

- Rainfall adequacy ratio.
- Rainfall difference.
- Rainfall per dry-spell day.
- Weather stress index.
- Total NPK.
- Nitrogen-to-phosphorus ratio.
- Potassium-to-nitrogen ratio.
- Soil pH distance from 6.5.
- Price-to-MSP ratio.
- Price premium over MSP.
- Log-transformed cultivated area.
- Previous-year district-crop yield.
- Rolling three-year district-crop yield.
- Previous-year district rainfall.

Historical yield features are district-crop aggregates, not farm-specific historical measurements.

---

## 6. Model Development

Models considered:

1. Baseline Mean Regressor.
2. Ridge Regression.
3. Random Forest Regressor.
4. LightGBM Regressor.

The final model selection should be based on validation performance and chronological cross-validation.

**Selected Model:** [Enter actual selected model]

**Hyperparameter Tuning:** [Enter actual tuning configuration]

---

## 7. Model Performance

Complete this section using actual evaluation results.

| Metric | Validation | Final Test (2024) |
|---|---|---|
| MAE (q/ha) | [Enter result] | [Enter result] |
| RMSE (q/ha) | [Enter result] | [Enter result] |
| R² | [Enter result] | [Enter result] |

**Baseline MAE:** [Enter actual baseline MAE]

**Improvement over baseline:** [Calculate using actual MAE values]

Do not report performance targets as achieved unless the results demonstrate them.

---

## 8. Model Explainability

SHAP analysis is used to understand the contribution of input features to model predictions.

**Top Five Features:**
1. [Feature 1]
2. [Feature 2]
3. [Feature 3]
4. [Feature 4]
5. [Feature 5]

SHAP values describe model behavior and do not establish causal relationships.

---

## 9. Limitations

The current project has the following limitations:

1. The dataset is synthetic rather than collected from actual farms.
2. Unit correction uses heuristic thresholds because explicit unit metadata is not available in the generated data.
3. Historical yield features represent district-crop averages, not individual farm histories.
4. Weather and soil patterns may not reflect real agricultural conditions.
5. The project does not establish that the model will generalize to other districts or years.
6. Predictions should not be interpreted as guaranteed crop production.
7. The model does not incorporate every factor affecting agricultural yield, such as irrigation access, pest outbreaks, seed varieties and farm management practices.

---

## 10. Fairness and Stability

Model performance should be examined separately for:

- Each district.
- Each crop.
- Each agricultural season.
- Different historical years.

Groups with higher prediction errors should be investigated before operational deployment.

The current synthetic dataset cannot establish fairness or reliability for real farming communities.

---

## 11. Ethical Considerations

- Do not use synthetic predictions as official agricultural estimates.
- Avoid making financial or procurement commitments solely from model outputs.
- Validate the model with real agricultural data before deployment.
- Communicate uncertainty and prediction limitations.
- Review performance across different districts and crops.

---

## 12. Conclusion

This project demonstrates an end-to-end machine learning workflow for district-level crop yield prediction using synthetic agricultural data.

It covers exploratory data analysis, data cleaning, feature engineering, chronological data splitting, model comparison, hyperparameter tuning, model evaluation and explainability.

The system serves as a prototype for agricultural analytics and requires validation using reliable real-world agricultural datasets before practical deployment.
