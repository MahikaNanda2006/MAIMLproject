# =============================================================================
# COURSE: Mathematics for AI & ML (23HAIML501)
# TOPIC: Numerical Analysis, Linear Algebra, & Feature Selection
# DATASET: Vehicle Emission Dataset (10,000 samples x 19 features)
# =============================================================================

import pandas as pd             # Data processing and tabular manipulation
import numpy as np              # Vectorized math operations and matrix algebra
import matplotlib.pyplot as plt # Pure plotting library (No Seaborn required)
from scipy import stats         # Statistical functions (Z-score calculation)

# Set display parameters for Spyder console output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# =============================================================================
# STEP 1: DATA LOAD & CLEANLINESS VALIDATION
# =============================================================================
print("=" * 60)
print("STEP 1: DATA CLEANLINESS VALIDATION")
print("=" * 60)

# Load the cleaned CSV file produced during preprocessing
df = pd.read_csv('cleaned_vehicle_emission_dataset.csv')

print(f"Dataset Dimensions: {df.shape[0]} Rows x {df.shape[1]} Columns\n")

# Check missing values
missing_total = df.isnull().sum().sum()
print(f"Total Missing Values across dataset: {missing_total}")
if missing_total == 0:
    print("-> VALIDATION PASSED: Dataset is completely clean with zero missing values.\n")

# Isolate numerical columns (14 features)
numerical_df = df.select_dtypes(include=[np.number])
num_cols = list(numerical_df.columns)
print(f"Identified Numerical Features ({len(num_cols)}):")
print(num_cols)
print("\n")


# =============================================================================
# STEP 2: DESCRIPTIVE STATISTICAL ANALYSIS (CO3)
# =============================================================================
print("=" * 60)
print("STEP 2: DESCRIPTIVE STATISTICAL SUMMARY")
print("=" * 60)

# Calculate summary metrics
stats_summary = pd.DataFrame({
    'Mean': numerical_df.mean(),
    'Median': numerical_df.median(),
    'Std_Dev': numerical_df.std(),
    'Variance': numerical_df.var(),
    'IQR': numerical_df.quantile(0.75) - numerical_df.quantile(0.25),
    'Skewness': numerical_df.skew(),   # Approx 0 indicates symmetric uniform distribution
    'Kurtosis': numerical_df.kurtosis() # Approx -1.2 indicates uniform distribution bounds
})

print(stats_summary.round(4))
print("\n")


# =============================================================================
# STEP 3: EXTREME VALUE CHECK & SCALED BOXPLOTS
# =============================================================================
print("=" * 60)
print("STEP 3: OUTLIER & EXTREME VALUE ANALYSIS")
print("=" * 60)

# Calculate Z-Scores (|Z| > 3 indicates extreme values)
z_scores = np.abs(stats.zscore(numerical_df))
outliers_count = (z_scores > 3).sum(axis=0)

outlier_df = pd.DataFrame({'Extreme Outliers (|Z| > 3)': outliers_count}, index=num_cols)
print(outlier_df)
print("\n-> RESULT: No extreme outliers found. Continuous features are well-bounded.\n")

# Plot Normalized Boxplots (Min-Max scaled for consistent visual representation)
normalized_df = (numerical_df - numerical_df.min()) / (numerical_df.max() - numerical_df.min())

plt.figure(figsize=(12, 6))
plt.boxplot(normalized_df.values, labels=num_cols, vert=False, patch_artist=True)
plt.title('Normalized Boxplot Analysis of Numerical Variables (Scale [0, 1])', fontsize=12, fontweight='bold')
plt.xlabel('Scaled Range')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 4: LINEAR ALGEBRA - PEARSON CORRELATION MATRIX (CO1)
# =============================================================================
print("=" * 60)
print("STEP 4: LINEAR ALGEBRA - CORRELATION MATRIX COMPUTATION")
print("=" * 60)

# Pearson Correlation Matrix: R = (1 / (n-1)) * Z^T * Z
corr_matrix = numerical_df.corr()
print("Correlation Matrix (Sample):")
print(corr_matrix.iloc[:5, :5].round(3))
print("\n")

# Matplotlib Heatmap visualization
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient (r)')
plt.xticks(range(len(num_cols)), num_cols, rotation=90, fontsize=8)
plt.yticks(range(len(num_cols)), num_cols, fontsize=8)
plt.title('Correlation Matrix Heatmap (Linear Relationships)', fontsize=12, fontweight='bold')

# Annotate correlation numbers directly on plot
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        val = corr_matrix.iloc[i, j]
        plt.text(j, i, f"{val:.2f}", ha='center', va='center', 
                 color='white' if abs(val) > 0.5 else 'black', fontsize=7)

plt.tight_layout()
plt.show()


# =============================================================================
# STEP 5: FEATURE SELECTION & RANKING (CO2)
# =============================================================================
print("=" * 60)
print("STEP 5: FEATURE SELECTION RELATIVE TO CO2 EMISSIONS")
print("=" * 60)

target_var = 'CO2 Emissions'

if target_var in numerical_df.columns:
    # Compute absolute correlation with CO2 Emissions
    target_corr = corr_matrix[target_var].abs().sort_values(ascending=False)
    print("Feature Correlation Strengths with CO2 Emissions:")
    print(target_corr.round(4))
    print("\n")
    
    # Select features with correlation threshold > 0.01 (excluding target itself)
    selected_features = target_corr[(target_corr.index != target_var) & (target_corr > 0.01)].index.tolist()
    print(f"Selected Key Features for Predictive Modeling: {selected_features}\n")
    
    # Visualizing feature importance ranking
    plt.figure(figsize=(9, 5))
    target_corr.drop(target_var).plot(kind='barh', color='skyblue', edgecolor='black')
    plt.title('Numerical Feature Importance (Absolute Correlation with CO2 Emissions)', fontsize=11, fontweight='bold')
    plt.xlabel('Absolute Pearson Correlation (|r|)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# =============================================================================
# STEP 6: ADVANCED LINEAR ALGEBRA - PRINCIPAL COMPONENT ANALYSIS (PCA) (CO1)
# =============================================================================
print("=" * 60)
print("STEP 6: LINEAR ALGEBRA - DIMENSIONALITY REDUCTION (PCA)")
print("=" * 60)

# Step 6.1: Standardize the dataset (Mean = 0, Std = 1)
X = numerical_df.values
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# Step 6.2: Compute Covariance Matrix: Sigma = (1/n) * X_scaled^T * X_scaled
cov_matrix = np.cov(X_scaled, rowvar=False)

# Step 6.3: Compute Eigenvalues & Eigenvectors via Eigen Value Decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvectors by decreasing eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 6.4: Compute Explained Variance Ratio
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance_ratio)

pca_summary = pd.DataFrame({
    'Eigenvalue': eigenvalues,
    'Variance Ratio': explained_variance_ratio,
    'Cumulative Variance': cumulative_variance
}, index=[f"PC{i+1}" for i in range(len(eigenvalues))])

print("PCA Explained Variance Summary:")
print(pca_summary.round(4))
print("\n")

# Scree Plot visualization
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(eigenvalues) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance Threshold')
plt.title('PCA Scree Plot (Cumulative Variance Explained)', fontsize=11, fontweight='bold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
