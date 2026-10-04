import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "outputs/cleaned_vehicle_emission_dataset.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("CATEGORICAL VARIABLE ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)


# ============================================================
# CATEGORICAL VARIABLES
# ============================================================

categorical_variables = [
    "Vehicle Type",
    "Fuel Type",
    "Road Type",
    "Traffic Conditions"
]

target = "CO2 Emissions"


# ============================================================
# 1. DISTRIBUTION OF CATEGORICAL VARIABLES
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL VARIABLE DISTRIBUTIONS")
print("=" * 60)

for column in categorical_variables:
    print(f"\n{column}:")
    print(df[column].value_counts())
    print()


# ============================================================
# 2. ONE-WAY ANOVA
# ============================================================

print("\n" + "=" * 60)
print("ONE-WAY ANOVA")
print("=" * 60)

anova_results = []

for column in categorical_variables:

    data = df[[column, target]].dropna()

    groups = [
        group[target].values
        for _, group in data.groupby(column)
    ]

    f_statistic, p_value = stats.f_oneway(*groups)

    anova_results.append({
        "Categorical Variable": column,
        "Target Variable": target,
        "F-statistic": f_statistic,
        "p-value": p_value,
        "Significant (alpha=0.05)": "Yes" if p_value < 0.05 else "No"
    })

    print(f"\n{column} -> {target}")

    print("\nGroup statistics:")
    print(
        data.groupby(column)[target]
        .agg(["count", "mean", "std"])
        .round(3)
    )

    print(f"\nF-statistic: {f_statistic:.4f}")
    print(f"p-value: {p_value:.6g}")

    if p_value < 0.05:
        print("Result: Significant difference between groups.")
    else:
        print("Result: No statistically significant difference between groups.")


# ============================================================
# SAVE ANOVA RESULTS
# ============================================================

anova_df = pd.DataFrame(anova_results)

anova_output = os.path.join(OUTPUT_DIR, "anova_results.csv")
anova_df.to_csv(anova_output, index=False)

print("\nANOVA results saved to:", anova_output)

# ============================================================
# 2B. TUKEY HSD POST-HOC TEST FOR SIGNIFICANT ANOVA
# ============================================================

from statsmodels.stats.multicomp import pairwise_tukeyhsd

print("\n" + "=" * 60)
print("TUKEY HSD POST-HOC TEST")
print("=" * 60)

road_data = df[["Road Type", "CO2 Emissions"]].dropna()

tukey_result = pairwise_tukeyhsd(
    endog=road_data["CO2 Emissions"],
    groups=road_data["Road Type"],
    alpha=0.05
)

print("\nTukey HSD: Road Type -> CO2 Emissions")
print(tukey_result)

# Save Tukey results
tukey_output = os.path.join(
    OUTPUT_DIR,
    "tukey_road_type_results.txt"
)

with open(tukey_output, "w") as f:
    f.write(str(tukey_result))

print("\nTukey results saved to:", tukey_output)

# ============================================================
# 2C. ANOVA ASSUMPTION CHECKS
# ============================================================

print("\n" + "=" * 60)
print("ANOVA ASSUMPTION CHECKS")
print("=" * 60)

for column in categorical_variables:

    data = df[[column, target]].dropna()

    groups = [
        group[target].values
        for _, group in data.groupby(column)
    ]

    # Levene's test for equality of variances
    levene_stat, levene_p = stats.levene(
        *groups,
        center="median"
    )

    print(f"\n{column} -> {target}")
    print(f"Levene statistic: {levene_stat:.4f}")
    print(f"Levene p-value: {levene_p:.6g}")

    if levene_p < 0.05:
        print("Result: Variances are significantly different.")
    else:
        print("Result: No significant evidence of unequal variances.")

# ============================================================
# 3. BOX PLOTS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING BOXPLOTS")
print("=" * 60)

for column in categorical_variables:

    plt.figure(figsize=(10, 6))

    df.boxplot(
        column=target,
        by=column
    )

    plt.title(f"CO2 Emissions by {column}")
    plt.suptitle("")
    plt.xlabel(column)
    plt.ylabel("CO2 Emissions")

    plt.xticks(rotation=30)
    plt.tight_layout()

    safe_name = column.replace(" ", "_")
    output_file = os.path.join(
        OUTPUT_DIR,
        f"boxplot_CO2_by_{safe_name}.png"
    )

    plt.savefig(output_file, dpi=300)
    plt.close()

    print("Saved:", output_file)


# ============================================================
# 4. EMISSION LEVEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("EMISSION LEVEL DISTRIBUTION")
print("=" * 60)

if "Emission Level" in df.columns:

    emission_counts = df["Emission Level"].value_counts()

    print(emission_counts)

    emission_counts.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "emission_level_distribution.csv"
        )
    )

print("\nCategorical analysis completed.")