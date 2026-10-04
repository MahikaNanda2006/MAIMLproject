"""
data_cleaning_preprocessing.py

Data cleaning and preprocessing pipeline for vehicle emission analysis.

Input:
    vehicle_emission_dataset.csv

Outputs:
    outputs/
        cleaned_vehicle_emission_dataset.csv
        processed_vehicle_emission_dataset.csv
        feature_data.csv
        target.csv
        outlier_report.csv
        preprocessing_summary.txt

The pipeline:
1. Loads the raw CSV.
2. Cleans column names and categorical text.
3. Converts numeric columns safely.
4. Replaces invalid/infinite values with NaN.
5. Handles missing values (median for numeric, mode for categorical).
6. Removes exact duplicate rows.
7. Performs basic domain/range validation.
8. Generates an IQR-based outlier report.
9. Optionally caps numerical outliers (disabled by default).
10. One-hot encodes categorical predictors.
11. Standardizes numerical predictors.
12. Encodes Emission Level as an ordinal target:
        Low = 0, Medium = 1, High = 2

Important:
- Outliers are NOT deleted by default. Emission data can naturally contain
  unusual observations, so the script reports them instead.
- The cleaned dataset is kept in human-readable form.
- The processed dataset is suitable for numerical/categorical modelling.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "vehicle_emission_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# Set this to True only if your analysis/model specifically requires
# outlier capping. By default, unusual observations are preserved.
CAP_OUTLIERS = False

# IQR multiplier used for outlier detection/capping.
IQR_MULTIPLIER = 1.5

TARGET_COLUMN = "Emission Level"


# ============================================================
# EXPECTED COLUMN GROUPS
# ============================================================

CATEGORICAL_COLUMNS = [
    "Vehicle Type",
    "Fuel Type",
    "Road Type",
    "Traffic Conditions",
]

NUMERICAL_COLUMNS = [
    "Engine Size",
    "Age of Vehicle",
    "Mileage",
    "Speed",
    "Acceleration",
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Air Pressure",
    "CO2 Emissions",
    "NOx Emissions",
    "PM2.5 Emissions",
    "VOC Emissions",
    "SO2 Emissions",
]

# Columns where negative values do not make physical sense.
NON_NEGATIVE_COLUMNS = [
    "Engine Size",
    "Age of Vehicle",
    "Mileage",
    "Speed",
    "Acceleration",
    "Humidity",
    "Wind Speed",
    "Air Pressure",
    "CO2 Emissions",
    "NOx Emissions",
    "PM2.5 Emissions",
    "VOC Emissions",
    "SO2 Emissions",
]

# Humidity is normally bounded between 0 and 100 percent.
BOUNDED_COLUMNS = {
    "Humidity": (0, 100),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip unnecessary whitespace from column names."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def clean_categorical_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize categorical strings without changing their meaning."""
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

            # Treat common textual representations of missing data as missing.
            df[col] = df[col].replace(
                {
                    "": pd.NA,
                    "NA": pd.NA,
                    "N/A": pd.NA,
                    "n/a": pd.NA,
                    "null": pd.NA,
                    "NULL": pd.NA,
                    "None": pd.NA,
                    "none": pd.NA,
                    "?": pd.NA,
                }
            )

    return df


def convert_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Safely convert expected numerical columns to numeric dtype."""
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn physically/semantically invalid values into NaN.

    They are then handled by the missing-value imputation step.
    """
    df = df.copy()

    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    for col, (lower, upper) in BOUNDED_COLUMNS.items():
        if col in df.columns:
            invalid = (df[col] < lower) | (df[col] > upper)
            df.loc[invalid, col] = np.nan

    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values.

    Numerical:
        median

    Categorical:
        mode

    Median is used for numerical variables because it is less sensitive
    to outliers than the mean.
    """
    df = df.copy()
    imputation_values = {}

    for col in numeric_columns:
        if col not in df.columns:
            continue

        missing_count = int(df[col].isna().sum())

        if missing_count > 0:
            median_value = df[col].median()

            # If a column is entirely missing, use 0 as a final fallback.
            if pd.isna(median_value):
                median_value = 0.0

            df[col] = df[col].fillna(median_value)
            imputation_values[col] = f"median = {median_value}"

    for col in categorical_columns:
        if col not in df.columns:
            continue

        missing_count = int(df[col].isna().sum())

        if missing_count > 0:
            modes = df[col].mode(dropna=True)

            # If the whole column is missing, use "Unknown".
            mode_value = modes.iloc[0] if not modes.empty else "Unknown"

            df[col] = df[col].fillna(mode_value)
            imputation_values[col] = f"mode = {mode_value}"

    return df, imputation_values


def create_outlier_report(
    df: pd.DataFrame,
    numeric_columns: list[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Create an IQR-based outlier report without deleting observations."""
    records = []

    for col in numeric_columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)

        records.append(
            {
                "Column": col,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower_bound,
                "Upper Bound": upper_bound,
                "Outlier Count": int(outlier_mask.sum()),
                "Outlier Percentage": float(outlier_mask.mean() * 100),
            }
        )

    return pd.DataFrame(records)


def cap_outliers(
    df: pd.DataFrame,
    numeric_columns: list[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Cap numerical values at IQR bounds (winsorization-style)."""
    df = df.copy()

    for col in numeric_columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def make_preprocessed_data(
    df: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical predictors and standardize numerical predictors.

    Returns:
        processed_features: encoded + scaled predictor matrix
        encoded_target: ordinal target as 0/1/2
    """
    feature_columns = [
        col for col in categorical_columns + numerical_columns
        if col in df.columns
    ]

    X = df[feature_columns].copy()

    # One-hot encode categorical variables.
    # handle_unknown='ignore' prevents errors when future data contains
    # a category not present in the original dataset.
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
    except TypeError:
        # Compatibility with older scikit-learn versions.
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float64,
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), [
                col for col in numerical_columns if col in X.columns
            ]),
            ("categorical", encoder, [
                col for col in categorical_columns if col in X.columns
            ]),
        ],
        remainder="drop",
    )

    processed_array = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    processed_features = pd.DataFrame(
        processed_array,
        columns=feature_names,
        index=df.index,
    )

    # Keep the target separate from the predictors.
    # This avoids accidentally leaking the target into feature preprocessing.
    target_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    encoded_target = (
        df[target_column]
        .map(target_mapping)
        .rename(target_column)
        .to_frame()
    )

    # If unexpected target labels exist, preserve them as missing rather
    # than silently assigning an incorrect numerical value.
    if encoded_target[target_column].isna().any():
        unknown_targets = sorted(
            df.loc[
                encoded_target[target_column].isna(),
                target_column
            ].astype(str).unique()
        )
        warnings.warn(
            f"Unexpected target labels found: {unknown_targets}. "
            "Their encoded values will be NaN."
        )

    return processed_features, encoded_target


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input dataset:\n{INPUT_FILE}\n\n"
            "Make sure vehicle_emission_dataset.csv is in the same "
            "folder as this script."
        )

    print("=" * 70)
    print("VEHICLE EMISSION DATA CLEANING & PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    df = pd.read_csv(INPUT_FILE)

    original_rows, original_columns = df.shape

    print(f"\nOriginal shape: {df.shape}")

    # --------------------------------------------------------
    # 2. Clean column names
    # --------------------------------------------------------
    df = clean_column_names(df)

    # Check that the expected target exists.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # 3. Clean categorical variables
    # --------------------------------------------------------
    df = clean_categorical_values(df, CATEGORICAL_COLUMNS + [TARGET_COLUMN])

    # --------------------------------------------------------
    # 4. Convert numerical variables
    # --------------------------------------------------------
    df = convert_numeric_columns(df, NUMERICAL_COLUMNS)

    # --------------------------------------------------------
    # 5. Replace infinite values
    # --------------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    # --------------------------------------------------------
    # 6. Remove exact duplicate rows
    # --------------------------------------------------------
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # --------------------------------------------------------
    # 7. Basic range/domain validation
    # --------------------------------------------------------
    df = validate_ranges(df)

    # --------------------------------------------------------
    # 8. Handle missing values
    # --------------------------------------------------------
    missing_before = int(df.isna().sum().sum())

    df, imputation_values = handle_missing_values(
        df,
        NUMERICAL_COLUMNS,
        CATEGORICAL_COLUMNS + [TARGET_COLUMN],
    )

    missing_after = int(df.isna().sum().sum())

    # --------------------------------------------------------
    # 9. Outlier analysis
    # --------------------------------------------------------
    outlier_report = create_outlier_report(
        df,
        NUMERICAL_COLUMNS,
        multiplier=IQR_MULTIPLIER,
    )

    outlier_report.to_csv(
        OUTPUT_DIR / "outlier_report.csv",
        index=False,
    )

    # Optional outlier capping.
    if CAP_OUTLIERS:
        df = cap_outliers(
            df,
            NUMERICAL_COLUMNS,
            multiplier=IQR_MULTIPLIER,
        )

    # --------------------------------------------------------
    # 10. Save cleaned human-readable dataset
    # --------------------------------------------------------
    cleaned_file = OUTPUT_DIR / "cleaned_vehicle_emission_dataset.csv"
    df.to_csv(cleaned_file, index=False)

    # --------------------------------------------------------
    # 11. Encode + scale features
    # --------------------------------------------------------
    processed_features, encoded_target = make_preprocessed_data(
        df,
        CATEGORICAL_COLUMNS,
        NUMERICAL_COLUMNS,
        TARGET_COLUMN,
    )

    # Combine processed predictors and encoded target.
    processed_dataset = pd.concat(
        [
            processed_features.reset_index(drop=True),
            encoded_target.reset_index(drop=True),
        ],
        axis=1,
    )

    processed_dataset.to_csv(
        OUTPUT_DIR / "processed_vehicle_emission_dataset.csv",
        index=False,
    )

    processed_features.to_csv(
        OUTPUT_DIR / "feature_data.csv",
        index=False,
    )

    encoded_target.to_csv(
        OUTPUT_DIR / "target.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 12. Generate summary report
    # --------------------------------------------------------
    summary_lines = [
        "VEHICLE EMISSION DATA CLEANING & PREPROCESSING SUMMARY",
        "=" * 60,
        f"Input file: {INPUT_FILE.name}",
        f"Original rows: {original_rows}",
        f"Original columns: {original_columns}",
        f"Final rows: {len(df)}",
        f"Final columns: {len(df.columns)}",
        f"Duplicate rows removed: {duplicate_count}",
        f"Missing values before imputation: {missing_before}",
        f"Missing values after imputation: {missing_after}",
        f"Outlier capping enabled: {CAP_OUTLIERS}",
        "",
        "Categorical columns:",
        *[f"  - {col}" for col in CATEGORICAL_COLUMNS],
        "",
        "Numerical columns:",
        *[f"  - {col}" for col in NUMERICAL_COLUMNS],
        "",
        "Target:",
        f"  - {TARGET_COLUMN}",
        "  - Low = 0",
        "  - Medium = 1",
        "  - High = 2",
        "",
        "Imputation performed:",
    ]

    if imputation_values:
        summary_lines.extend(
            [f"  - {col}: {value}" for col, value in imputation_values.items()]
        )
    else:
        summary_lines.append("  - No missing values required imputation.")

    summary_lines.extend(
        [
            "",
            "Output files:",
            "  - cleaned_vehicle_emission_dataset.csv",
            "  - processed_vehicle_emission_dataset.csv",
            "  - feature_data.csv",
            "  - target.csv",
            "  - outlier_report.csv",
            "  - preprocessing_summary.txt",
        ]
    )

    with open(OUTPUT_DIR / "preprocessing_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # --------------------------------------------------------
    # 13. Console summary
    # --------------------------------------------------------
    print(f"Rows after cleaning: {len(df)}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Missing values before imputation: {missing_before}")
    print(f"Missing values after imputation: {missing_after}")
    print(f"Processed feature shape: {processed_features.shape}")

    print("\nOutput files created in:")
    print(OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
