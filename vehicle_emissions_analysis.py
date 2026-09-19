import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

import os

#1. Load dataset
df = pd.read_csv("vehicle_emission_dataset.csv")

print("Dataset Loaded")
print("Dimensions:" , df.shape[0],df.shape[1])
print("Column names: ")
print(df.columns.tolist())
print("")

#2. Basic dataset information

print("Dataset information")
print("")
print("Data types: ")
print(df.dtypes)

print("")
print("Missing values:")
print(df.isnull().sum())

print("\nDuplicate rows: ", df.duplicated().sum())
print("\nFirst five rows: ")
print(df.head())
print("\nLast five rows: ")
print(df.tail())

print("\nStatistical Summary:")
print(df.describe())

#Output work
os.makedirs("outputs", exist_ok = True)
print("\nOutput directory ready")   

#3. Identify Numerical and categorical features


print("NUMERICAL AND CATEGORICAL FEATURES")

numerical_features = df.select_dtypes(include=np.number).columns.tolist()
categorical_features = df.select_dtypes(include="object").columns.tolist()

print("\nNumerical features:")
for feature in numerical_features:
    print("-", feature)

print("\nCategorical features:")
for feature in categorical_features:
    print("-", feature)

print("\nNumber of numerical features:", len(numerical_features))
print("Number of categorical features:", len(categorical_features))

# 4. Categorical Variable Analysis

print("CATEGORICAL VARIABLE ANALYSIS")

for feature in categorical_features:
    print("\n" + feature)
    print("-" * len(feature))
    print(df[feature].value_counts())

    # ============================================================
# 5. VISUALIZATION OF CATEGORICAL VARIABLES
# ============================================================

for feature in categorical_features:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=feature
    )

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of Vehicles")
    plt.xticks(rotation=30)

    plt.tight_layout()

    filename = feature.replace(" ", "_") + "_distribution.png"

    plt.savefig(
        f"outputs/{filename}",
        dpi=300
    )

    plt.close()

    # ============================================================
# 6. CO2 EMISSIONS ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CO2 EMISSIONS ANALYSIS")
print("=" * 60)

print("\nCO2 Emissions Statistics:")
print(df["CO2 Emissions"].describe())

plt.figure(figsize=(9, 5))

sns.histplot(
    df["CO2 Emissions"],
    bins=30,
    kde=True
)

plt.title("Distribution of CO2 Emissions")
plt.xlabel("CO2 Emissions")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/co2_distribution.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(9, 4))

sns.boxplot(
    x=df["CO2 Emissions"]
)

plt.title("Box Plot of CO2 Emissions")
plt.xlabel("CO2 Emissions")

plt.tight_layout()

plt.savefig(
    "outputs/co2_boxplot.png",
    dpi=300
)

plt.close()

# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

numeric_df = df.select_dtypes(include=np.number)

correlation_matrix = numeric_df.corr()

print("\nCorrelation with CO2 Emissions:")

co2_correlations = (
    correlation_matrix["CO2 Emissions"]
    .sort_values(ascending=False)
)

print(co2_correlations)

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Numerical Variables")

plt.tight_layout()

plt.savefig(
    "outputs/correlation_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# CO2 FEATURE CORRELATIONS
# ============================================================

co2_feature_correlations = (
    correlation_matrix["CO2 Emissions"]
    .drop("CO2 Emissions")
    .sort_values()
)

plt.figure(figsize=(9, 7))

co2_feature_correlations.plot(
    kind="barh"
)

plt.title("Correlation of Numerical Features with CO2 Emissions")
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "outputs/co2_feature_correlations.png",
    dpi=300
)

plt.close()

print("\nCorrelation of features with CO2:")
print(co2_feature_correlations)

# ============================================================
# 8. PEARSON CORRELATION AND P-VALUES
# ============================================================

print("\n" + "=" * 60)
print("PEARSON CORRELATION AND STATISTICAL SIGNIFICANCE")
print("=" * 60)

# CO2 is the target, so don't include it as a predictor.
# Emission Level is excluded because it represents an
# emission category and may cause target leakage.

candidate_features = [
    feature
    for feature in numerical_features
    if feature != "CO2 Emissions"
    and feature != "Emission Level"
]

results = []

for feature in candidate_features:

    r, p = stats.pearsonr(
        df[feature],
        df["CO2 Emissions"]
    )

    results.append({
        "Feature": feature,
        "Correlation": r,
        "P-value": p
    })

correlation_results = pd.DataFrame(results)

correlation_results["Absolute Correlation"] = (
    correlation_results["Correlation"].abs()
)

correlation_results = correlation_results.sort_values(
    by="Absolute Correlation",
    ascending=False
)

print(correlation_results.to_string(index=False))

# ============================================================
# 9. FEATURE VS CO2 SCATTER PLOTS
# ============================================================

for feature in candidate_features:

    plt.figure(figsize=(7, 5))

    plt.scatter(
        df[feature],
        df["CO2 Emissions"],
        alpha=0.3
    )

    plt.xlabel(feature)
    plt.ylabel("CO2 Emissions")
    plt.title(f"{feature} vs CO2 Emissions")

    plt.tight_layout()

    filename = (
        feature.replace(" ", "_")
        .replace(".", "")
        .replace("/", "_")
        + "_vs_CO2.png"
    )

    plt.savefig(
        f"outputs/{filename}",
        dpi=300
    )

    plt.close()