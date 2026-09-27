# ============================================================
# VEHICLE EMISSION ANALYSIS
# Covers the T1 rubric:
# 1. Linear algebra + SVD/PCA + visualization
# 2. Feature selection
# 3. Statistical analysis on multivariate data
# 4. Machine-learning regression models for CO2 prediction
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

# -------------------- 1. LOAD DATA --------------------
DATA_FILE = "vehicle_emission_dataset.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("VEHICLE EMISSION ANALYSIS")
print("=" * 70)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nFirst five rows:")
print(df.head())

print("\nStatistical summary:")
print(df.describe())

# The target is CO2 Emissions.
TARGET = "CO2 Emissions"

# We do NOT use other pollutant outputs or Emission Level as predictors.
# This avoids using information that is essentially another emission output
# or a category derived from the target.
NUMERIC_FEATURES = [
    "Engine Size",
    "Age of Vehicle",
    "Mileage",
    "Speed",
    "Acceleration",
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Air Pressure"
]

CATEGORICAL_FEATURES = [
    "Vehicle Type",
    "Fuel Type",
    "Road Type",
    "Traffic Conditions"
]

# -------------------- 2. BASIC VISUAL ANALYSIS --------------------
print("\n" + "=" * 70)
print("VISUAL ANALYSIS")
print("=" * 70)

plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET], bins=30, kde=True)
plt.title("Distribution of CO2 Emissions")
plt.xlabel("CO2 Emissions")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_co2_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df[TARGET])
plt.title("Box Plot of CO2 Emissions")
plt.xlabel("CO2 Emissions")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_co2_boxplot.png", dpi=300)
plt.close()

for feature in CATEGORICAL_FEATURES:
    plt.figure(figsize=(8, 5))
    order = df[feature].value_counts().index
    sns.countplot(data=df, x=feature, order=order)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of Vehicles")
    plt.xticks(rotation=30)
    plt.tight_layout()
    safe_name = feature.lower().replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/03_{safe_name}.png", dpi=300)
    plt.close()

# -------------------- 3. STATISTICAL ANALYSIS --------------------
print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

correlation_results = []

for feature in NUMERIC_FEATURES:
    r, p = stats.pearsonr(df[feature], df[TARGET])
    correlation_results.append({
        "Feature": feature,
        "Pearson_r": r,
        "P_value": p,
        "Absolute_r": abs(r)
    })

correlation_results = pd.DataFrame(correlation_results)
correlation_results = correlation_results.sort_values(
    "Absolute_r", ascending=False
)

print("\nPearson correlation and p-values:")
print(correlation_results.to_string(index=False))

correlation_results.to_csv(
    f"{OUTPUT_DIR}/correlation_pvalues.csv", index=False
)

# Correlation matrix
corr_df = df[NUMERIC_FEATURES + [TARGET]].corr()

plt.figure(figsize=(11, 8))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_matrix.png", dpi=300)
plt.close()

# Feature-vs-target plots for visual analysis
for feature in NUMERIC_FEATURES:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feature], df[TARGET], alpha=0.35)
    plt.xlabel(feature)
    plt.ylabel(TARGET)
    plt.title(f"{feature} vs CO2 Emissions")
    plt.tight_layout()
    safe_name = feature.lower().replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/05_{safe_name}_vs_co2.png", dpi=300)
    plt.close()

# -------------------- 4. FEATURE SELECTION --------------------
print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

X_numeric = df[NUMERIC_FEATURES]
y = df[TARGET]

# Standardize before SelectKBest.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Select the five strongest numerical features according to the
# univariate F-test.
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X_scaled, y)

selected_mask = selector.get_support()
selected_features = list(X_numeric.columns[selected_mask])

feature_scores = pd.DataFrame({
    "Feature": NUMERIC_FEATURES,
    "F_score": selector.scores_,
    "Selected": selected_mask
}).sort_values("F_score", ascending=False)

print("\nFeature-selection results:")
print(feature_scores.to_string(index=False))
print("\nSelected numerical features:", selected_features)

feature_scores.to_csv(
    f"{OUTPUT_DIR}/feature_selection_scores.csv", index=False
)

# -------------------- 5. LINEAR ALGEBRA: MATRIX OPERATIONS --------------------
print("\n" + "=" * 70)
print("LINEAR ALGEBRA: MATRIX OPERATIONS")
print("=" * 70)

# Z is the standardized numerical feature matrix.
Z = X_scaled

print("\nShape of standardized feature matrix Z:", Z.shape)

# Matrix multiplication:
# Z^T Z gives the unscaled covariance-related matrix.
ZTZ = Z.T @ Z

print("\nShape of Z^T Z:", ZTZ.shape)

# Covariance matrix:
covariance_matrix = ZTZ / (Z.shape[0] - 1)

print("\nCovariance matrix:")
print(pd.DataFrame(
    covariance_matrix,
    index=NUMERIC_FEATURES,
    columns=NUMERIC_FEATURES
).round(3))

# Singular Value Decomposition:
# Z = U S V^T
U, singular_values, VT = np.linalg.svd(Z, full_matrices=False)

print("\nSingular values:")
print(np.round(singular_values, 3))

# -------------------- 6. PCA USING SVD --------------------
print("\n" + "=" * 70)
print("PCA USING SVD")
print("=" * 70)

pca = PCA()
pca.fit(Z)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

pca_table = pd.DataFrame({
    "Principal_Component": [
        f"PC{i+1}" for i in range(len(explained))
    ],
    "Explained_Variance": explained,
    "Cumulative_Variance": cumulative
})

print("\nPCA explained variance:")
print(pca_table.to_string(index=False))

pca_table.to_csv(
    f"{OUTPUT_DIR}/pca_explained_variance.csv", index=False
)

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(explained) + 1),
    cumulative,
    marker="o"
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_pca_variance.png", dpi=300)
plt.close()

# -------------------- 7. MACHINE LEARNING --------------------
print("\n" + "=" * 70)
print("MACHINE LEARNING REGRESSION")
print("=" * 70)

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            drop="first"
        ), CATEGORICAL_FEATURES)
    ]
)

models = {
    "Multiple Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
    "Ridge Regression": Ridge(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=8
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=12,
        n_jobs=-1
    )
}

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)

    r2 = r2_score(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))

    results.append({
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })

    trained_models[name] = pipeline

results_df = pd.DataFrame(results)

print("\nModel comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    f"{OUTPUT_DIR}/model_comparison.csv", index=False
)

# Plot model R2 values.
plt.figure(figsize=(9, 5))
sns.barplot(data=results_df, x="R2", y="Model")
plt.title("Regression Model R² Comparison")
plt.xlabel("R² Score")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_model_comparison.png", dpi=300)
plt.close()

# -------------------- 8. FINAL SUMMARY --------------------
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nSelected features:")
for feature in selected_features:
    print("-", feature)

print("\nPCA cumulative variance:")
for i, value in enumerate(cumulative, start=1):
    print(f"PC{i}: {value:.4f}")

print("\nAll outputs have been saved in the 'outputs' folder.")
print("\nAnalysis completed successfully.")