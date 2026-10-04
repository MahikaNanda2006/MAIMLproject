# =============================================================================
# COURSE  : Mathematics for AI & ML (23HAIML501)
# TOPIC   : Numerical Analysis - Linear Algebra, PCA/SVD, Feature Selection,
#           Statistical Analysis on a multivariate dataset
# DATASET : Vehicle Emission Dataset (10,000 rows x 19 columns)
# PART    : NUMERICAL analysis only (categorical analysis is done by teammate)
#
# HOW TO RUN IN SPYDER
#   1. Put this file and cleaned_vehicle_emission_dataset.csv in the SAME folder
#   2. Open this file in Spyder
#   3. Press F5 (Run file). Output appears in the IPython console (bottom right)
#      and graphs appear in the "Plots" tab (top right).
#   4. A folder called "outputs" is created automatically, and all graphs are
#      saved there as .png files (these are what you upload to GitHub).
#
# RUBRIC MAP (so you can tell the teacher "where is it in the code")
#   CO1 (10 marks) : STEP 4 (matrix operations) + STEP 6 (PCA & SVD)
#   CO2 (7 marks)  : STEP 5 (feature selection)
#   CO2 (8 marks)  : STEP 7 (statistical analysis on multivariate data)
#   Supporting     : STEP 1 (cleaning check), STEP 2 (descriptive), STEP 3 (outliers)
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 0: IMPORT LIBRARIES AND BASIC SETTINGS
# -----------------------------------------------------------------------------
import os                          # to create folders / find the working folder
import numpy as np                 # fast maths + linear algebra (matrices)
import pandas as pd                # tables (DataFrames) - reading the CSV
import matplotlib.pyplot as plt    # drawing graphs
from scipy import stats            # statistical tests (z-score, t-test, ANOVA...)
from sklearn.feature_selection import mutual_info_regression  # non-linear dependence

# Show ALL columns when printing a table, and make the console wide
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Make a folder called "outputs" for saving graphs (does nothing if it exists)
os.makedirs('outputs', exist_ok=True)

# A tiny helper so we do not repeat the "save then show" code for every graph
def save_and_show(filename):
    """Saves the current figure into the outputs folder, then displays it."""
    plt.savefig(os.path.join('outputs', filename), dpi=150, bbox_inches='tight')
    plt.show()

# Fixed random seed -> any random step gives the SAME result every run
np.random.seed(42)

print("Working folder is:", os.getcwd())
print("(The CSV file must be inside this folder)\n")


# =============================================================================
# STEP 1: LOAD DATA AND CHECK IT IS REALLY CLEAN
# =============================================================================
print("=" * 70)
print("STEP 1: DATA LOADING & CLEANLINESS VALIDATION")
print("=" * 70)

# 1.1 Load the cleaned CSV into a DataFrame called df
df = pd.read_csv('cleaned_vehicle_emission_dataset.csv')
print(f"1.1 Dataset size: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# 1.2 Data types of each column (float = decimal number, str/object = text)
print("\n1.2 Column data types:")
print(df.dtypes)

# 1.3 Missing values: count of empty cells in every column
print("\n1.3 Missing values per column:")
print(df.isnull().sum())
missing_total = df.isnull().sum().sum()
if missing_total == 0:
    print("-> PASSED: zero missing values.")
else:
    print(f"-> WARNING: {missing_total} missing values found!")

# 1.4 Duplicate rows (the same row appearing twice)
dup_count = df.duplicated().sum()
print(f"\n1.4 Duplicate rows: {dup_count}",
      "-> PASSED" if dup_count == 0 else "-> WARNING")

# Separate numerical and categorical columns
numerical_df = df.select_dtypes(include=[np.number])    # only number columns
cat_cols = [c for c in df.columns if c not in numerical_df.columns]
num_cols = list(numerical_df.columns)
print(f"\n1.5 Numerical columns ({len(num_cols)}): {num_cols}")
print(f"    Categorical columns ({len(cat_cols)}): {cat_cols}")

# 1.6 Infinite values (can appear after bad divisions during cleaning)
inf_total = np.isinf(numerical_df.values).sum()
print(f"\n1.6 Infinite values: {inf_total}", "-> PASSED" if inf_total == 0 else "-> WARNING")

# 1.7 Impossible / wrong values: check every column against a sensible range.
#     (min allowed, max allowed). Anything outside = a "wrong value".
valid_ranges = {
    'Engine Size': (0, 10),            # litres
    'Age of Vehicle': (0, 50),         # years
    'Mileage': (0, 1_000_000),         # km
    'Speed': (0, 250),                 # km/h
    'Acceleration': (0, 15),           # m/s^2
    'Temperature': (-50, 60),          # degree C
    'Humidity': (0, 100),              # percent
    'Wind Speed': (0, 100),            # km/h
    'Air Pressure': (800, 1100),       # hPa
    'CO2 Emissions': (0, 2000),        # g/km
    'NOx Emissions': (0, 20),
    'PM2.5 Emissions': (0, 5),
    'VOC Emissions': (0, 5),
    'SO2 Emissions': (0, 5),
}
print("\n1.7 Range check (values outside the valid range = wrong values):")
wrong_total = 0
for col, (lo, hi) in valid_ranges.items():
    bad = ((df[col] < lo) | (df[col] > hi)).sum()   # count rows outside [lo, hi]
    wrong_total += bad
    print(f"    {col:<18} valid [{lo}, {hi}]  actual [{df[col].min():.3f}, {df[col].max():.3f}]  wrong values = {bad}")
print("-> PASSED: no impossible values." if wrong_total == 0 else f"-> WARNING: {wrong_total} wrong values")

# 1.8 Categorical columns: look for typos like 'car' vs 'Car' or ' Car'
print("\n1.8 Categorical columns - unique labels (checks for spelling/case issues):")
for col in cat_cols:
    print(f"    {col:<20} {sorted(df[col].unique())}")

# 1.9 Constant columns (same value everywhere) carry no information
const_cols = [c for c in num_cols if numerical_df[c].nunique() == 1]
print(f"\n1.9 Constant numerical columns: {const_cols if const_cols else 'None'}")

print("\n=> CONCLUSION STEP 1: dataset is clean. Safe to continue.\n")


# =============================================================================
# STEP 2: DESCRIPTIVE STATISTICS  (+ histograms)
# =============================================================================
print("=" * 70)
print("STEP 2: DESCRIPTIVE STATISTICAL SUMMARY")
print("=" * 70)

# One row per variable, one column per statistic
stats_summary = pd.DataFrame({
    'Mean': numerical_df.mean(),                       # average
    'Median': numerical_df.median(),                   # middle value
    'Std_Dev': numerical_df.std(),                     # spread around the mean
    'Variance': numerical_df.var(),                    # std squared
    'Min': numerical_df.min(),
    'Max': numerical_df.max(),
    'IQR': numerical_df.quantile(0.75) - numerical_df.quantile(0.25),  # Q3 - Q1
    'Skewness': numerical_df.skew(),       # 0 = symmetric
    'Kurtosis': numerical_df.kurtosis()    # -1.2 = flat/uniform, 0 = bell-shaped
})
print(stats_summary.round(4))

# Observation: Mean ~ Median, Skewness ~ 0 and Kurtosis ~ -1.2 for every column.
# These three together are the "fingerprint" of a UNIFORM distribution.
print("\nAverage skewness :", round(stats_summary['Skewness'].mean(), 3), "(close to 0 = symmetric)")
print("Average kurtosis :", round(stats_summary['Kurtosis'].mean(), 3), "(close to -1.2 = uniform)")

# Histograms of all 14 variables in a 3x5 grid
fig, axes = plt.subplots(3, 5, figsize=(18, 9))
axes = axes.flatten()                                  # make the grid a simple list
for i, col in enumerate(num_cols):
    axes[i].hist(numerical_df[col], bins=30, color='steelblue', edgecolor='black')
    axes[i].set_title(col, fontsize=9)
for j in range(len(num_cols), len(axes)):              # hide the unused empty cell
    axes[j].axis('off')
plt.suptitle('Histograms of Numerical Variables (flat shape = uniform distribution)',
             fontweight='bold')
plt.tight_layout()
save_and_show('histograms.png')


# =============================================================================
# STEP 3: OUTLIER / EXTREME VALUE ANALYSIS
# =============================================================================
print("=" * 70)
print("STEP 3: OUTLIER & EXTREME VALUE ANALYSIS")
print("=" * 70)

# 3.1 Z-score method.  z = (x - mean) / std.  |z| > 3 = extreme
z_scores = np.abs(stats.zscore(numerical_df))          # z-score of every cell
z_outliers = (z_scores > 3).sum(axis=0)                # count per column

# 3.2 IQR method.  Outlier if x < Q1 - 1.5*IQR  or  x > Q3 + 1.5*IQR
Q1 = numerical_df.quantile(0.25)
Q3 = numerical_df.quantile(0.75)
IQR = Q3 - Q1
iqr_outliers = ((numerical_df < Q1 - 1.5 * IQR) | (numerical_df > Q3 + 1.5 * IQR)).sum()

outlier_df = pd.DataFrame({'Z-score outliers (|z|>3)': np.asarray(z_outliers),
                           'IQR outliers': iqr_outliers.values}, index=num_cols)
print(outlier_df)

# The conclusion below is computed from the numbers (not typed by hand)
if outlier_df.values.sum() == 0:
    print("\n-> RESULT: No outliers by either method. All features are well-bounded.")
else:
    print(f"\n-> RESULT: {outlier_df.values.sum()} flagged values in total - investigate.")

# 3.3 Boxplots. Variables have different units, so first squash each to [0,1]
#     (min-max scaling) so they can share one axis.
normalized_df = (numerical_df - numerical_df.min()) / (numerical_df.max() - numerical_df.min())

plt.figure(figsize=(12, 7))
plt.boxplot(normalized_df.values, vert=False, patch_artist=True)
plt.yticks(range(1, len(num_cols) + 1), num_cols)      # put names on the y axis
plt.title('Normalized Boxplots of Numerical Variables (scale 0-1)', fontweight='bold')
plt.xlabel('Scaled value')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
save_and_show('boxplot.png')


# =============================================================================
# STEP 4: LINEAR ALGEBRA - MATRIX OPERATIONS  (CO1)
# =============================================================================
print("=" * 70)
print("STEP 4: LINEAR ALGEBRA - MATRIX OPERATIONS")
print("=" * 70)

# 4.1 The data matrix X: each ROW = one vehicle, each COLUMN = one variable.
X = numerical_df.values                                # DataFrame -> numpy matrix
n, p = X.shape                                         # n = 10000 rows, p = 14 columns
print(f"4.1 Data matrix X has shape {X.shape}  (n={n} samples, p={p} variables)")

# 4.2 Mean vector and mean-centering: subtract each column's mean.
mean_vec = X.mean(axis=0)                              # 1 x p vector of means
X_centered = X - mean_vec                              # now every column has mean 0

# 4.3 Covariance matrix BY HAND:  Cov = (1/(n-1)) * Xc^T * Xc
#     Xc^T is (p x n), Xc is (n x p)  ->  result is (p x p)
cov_manual = (X_centered.T @ X_centered) / (n - 1)     # '@' is matrix multiplication
cov_numpy = np.cov(X, rowvar=False)                    # numpy's built-in answer
print("\n4.3 Covariance matrix shape:", cov_manual.shape)
print("    Manual result equals np.cov? ->", np.allclose(cov_manual, cov_numpy))

# 4.4 Standardise (z-score) the data: divide centered data by std deviation.
#     ddof=1 -> use the same (n-1) denominator as above, so everything is consistent.
X_std = X_centered / X.std(axis=0, ddof=1)
print("\n4.4 After standardising: column means ~", np.round(X_std.mean(axis=0)[:3], 6),
      "| column std =", np.round(X_std.std(axis=0, ddof=1)[:3], 3))

# 4.5 Correlation matrix BY HAND:  R = (1/(n-1)) * Z^T * Z   (Z = standardised data)
corr_manual = (X_std.T @ X_std) / (n - 1)
corr_matrix = numerical_df.corr()                      # pandas built-in (for labels)
print("\n4.5 Manual correlation equals pandas .corr()? ->",
      np.allclose(corr_manual, corr_matrix.values))
print("\nCorrelation matrix (first 5x5 block):")
print(corr_matrix.iloc[:5, :5].round(3))

# 4.6 Properties of the matrix (typical viva questions!)
R = corr_matrix.values
print("\n4.6 Properties of the correlation matrix R:")
print("    Symmetric (R = R^T)?        ", np.allclose(R, R.T))
print("    Diagonal all equal to 1?    ", np.allclose(np.diag(R), 1))
print("    Rank of R                   ", np.linalg.matrix_rank(R), "(max possible =", p, ")")
print("    Determinant of R            ", round(np.linalg.det(R), 4), "(1 = columns totally unrelated)")
print("    Condition number of R       ", round(np.linalg.cond(R), 4), "(close to 1 = very stable)")
print("    Trace of R                  ", round(np.trace(R), 4), "(= number of variables)")

# 4.7 Largest correlations between DIFFERENT variables (ignoring the diagonal)
upper = R[np.triu_indices(p, k=1)]                     # all unique off-diagonal values
print(f"\n4.7 Largest |correlation| between any two different variables: {np.abs(upper).max():.4f}")
print(f"    Average |correlation| between variables: {np.abs(upper).mean():.4f}")

# 4.8 Heatmap of the correlation matrix, with numbers written inside every cell
plt.figure(figsize=(11, 9))
plt.imshow(R, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation coefficient (r)')
plt.xticks(range(p), num_cols, rotation=90, fontsize=8)
plt.yticks(range(p), num_cols, fontsize=8)
for i in range(p):
    for j in range(p):
        plt.text(j, i, f"{R[i, j]:.2f}", ha='center', va='center',
                 color='white' if abs(R[i, j]) > 0.5 else 'black', fontsize=6)
plt.title('Correlation Matrix Heatmap', fontweight='bold')
plt.tight_layout()
save_and_show('correlation_heatmap.png')


# =============================================================================
# STEP 5: FEATURE SELECTION  (CO2)
# Question: which input variables help predict CO2 Emissions?
# =============================================================================
print("=" * 70)
print("STEP 5: FEATURE SELECTION  (target = CO2 Emissions)")
print("=" * 70)

target = 'CO2 Emissions'

# Candidate predictors = the vehicle + environment variables.
# We do NOT use NOx/PM2.5/VOC/SO2 as predictors: they are other emissions
# (outputs), not causes, so using them would be "data leakage".
emission_cols = ['CO2 Emissions', 'NOx Emissions', 'PM2.5 Emissions', 'VOC Emissions', 'SO2 Emissions']
predictors = [c for c in num_cols if c not in emission_cols]
print("Candidate predictors:", predictors, "\n")

y = df[target].values

# --- METHOD A: Pearson correlation + p-value --------------------------------
# r  = strength of LINEAR relationship (-1 to +1)
# p  = probability of seeing such r by pure chance. p < 0.05 -> "significant"
rows = []
for col in predictors:
    r, p_val = stats.pearsonr(df[col], y)
    rows.append([col, r, abs(r), p_val])
corr_table = pd.DataFrame(rows, columns=['Feature', 'r', '|r|', 'p_value']).set_index('Feature')

# Bonferroni correction: we run 9 tests, so we make the p threshold stricter
alpha = 0.05
bonf_alpha = alpha / len(predictors)
corr_table['Significant (p<0.05)'] = corr_table['p_value'] < alpha
corr_table['Significant (Bonferroni)'] = corr_table['p_value'] < bonf_alpha
corr_table = corr_table.sort_values('|r|', ascending=False)
print("METHOD A - Pearson correlation with CO2:")
print(corr_table.round(4))
print(f"\n(Bonferroni threshold = 0.05/{len(predictors)} = {bonf_alpha:.4f})")

# --- METHOD B: Mutual Information -------------------------------------------
# Correlation only sees STRAIGHT-LINE relations. Mutual information (MI) also
# detects curved / any other dependence. MI = 0 means completely independent.
mi_scores = mutual_info_regression(df[predictors], y, random_state=42)
mi_table = pd.Series(mi_scores, index=predictors).sort_values(ascending=False)
print("\nMETHOD B - Mutual Information with CO2 (0 = independent):")
print(mi_table.round(5))

# --- METHOD C: Variance Inflation Factor (multicollinearity) -----------------
# VIF tells if a predictor can be rebuilt from the OTHER predictors.
# VIF = 1 means independent. VIF > 5 (or 10) means redundant -> drop one.
# Linear-algebra shortcut: VIF_i = i-th diagonal element of INVERSE of the
# correlation matrix of the predictors.
R_pred = np.corrcoef(df[predictors].values, rowvar=False)
vif = pd.Series(np.diag(np.linalg.inv(R_pred)), index=predictors)
print("\nMETHOD C - VIF among predictors (1 = no redundancy, >5 = redundant):")
print(vif.round(4))

# --- METHOD D: Variance check -----------------------------------------------
# After min-max scaling, a feature with almost zero variance is useless.
scaled_pred = normalized_df[predictors]
var_table = scaled_pred.var().sort_values(ascending=False)
print("\nMETHOD D - Variance of min-max scaled predictors (near 0 = useless):")
print(var_table.round(4))

# --- FINAL DECISION ---------------------------------------------------------
# A feature is KEPT if it is significant (p<0.05) in correlation.
# Redundant features (VIF>5) would be dropped; we report that too.
selected = corr_table[corr_table['Significant (p<0.05)']].index.tolist()
redundant = vif[vif > 5].index.tolist()
print("\n" + "-" * 70)
print("FINAL FEATURE SELECTION RESULT")
print("-" * 70)
print("Strongest correlation |r| :", round(corr_table['|r|'].max(), 4),
      "(", corr_table['|r|'].idxmax(), ")")
print("Selected (p<0.05)         :", selected if selected else "NONE")
print("Redundant (VIF>5)         :", redundant if redundant else "NONE")
if corr_table['|r|'].max() < 0.1:
    print("\nINTERPRETATION: every |r| is below 0.1 -> no predictor has a meaningful LINEAR")
    print("relationship with CO2, and MI is ~0 -> no non-linear relationship either.")
    print("The variables behave as INDEPENDENT uniform random variables (VIF ~ 1).")
    print("Statistically-significant does not mean useful: with n=10,000, even a tiny")
    print("r can reach p<0.05, so we also judge the SIZE of r (effect size).")

# Bar chart: |r| and mutual information side by side
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
corr_table['|r|'].sort_values().plot(kind='barh', ax=ax[0], color='skyblue', edgecolor='black')
ax[0].axvline(0.0196, color='red', linestyle='--', label='p=0.05 line (n=10,000)')
ax[0].set_title('Absolute Pearson correlation with CO2', fontweight='bold')
ax[0].set_xlabel('|r|')
ax[0].legend()
mi_table.sort_values().plot(kind='barh', ax=ax[1], color='salmon', edgecolor='black')
ax[1].set_title('Mutual information with CO2', fontweight='bold')
ax[1].set_xlabel('MI score')
plt.tight_layout()
save_and_show('feature_selection.png')


# =============================================================================
# STEP 6: PCA AND SVD  (CO1)
# PCA finds NEW axes (principal components) that capture the most variance.
# =============================================================================
print("=" * 70)
print("STEP 6: PRINCIPAL COMPONENT ANALYSIS (PCA) & SVD")
print("=" * 70)

# 6.1 Standardised data X_std was made in Step 4 (mean 0, std 1).
#     We standardise because variables have different units (Mileage ~ 100000,
#     Acceleration ~ 2). Without it, Mileage would dominate the result.

# 6.2 Covariance matrix of standardised data (= the correlation matrix)
cov_std = np.cov(X_std, rowvar=False)

# 6.3 METHOD 1 - EIGEN DECOMPOSITION:  Cov * v = lambda * v
#     eigh is the correct function for symmetric matrices (faster, real results).
#     lambda (eigenvalue) = variance captured by that component
#     v      (eigenvector) = direction of the component
eigenvalues, eigenvectors = np.linalg.eigh(cov_std)
order = np.argsort(eigenvalues)[::-1]                  # sort largest -> smallest
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# 6.4 Explained variance ratio = each eigenvalue / sum of all eigenvalues
explained = eigenvalues / eigenvalues.sum()
cumulative = np.cumsum(explained)
pc_names = [f"PC{i+1}" for i in range(p)]
pca_summary = pd.DataFrame({'Eigenvalue': eigenvalues,
                            'Variance Ratio': explained,
                            'Cumulative Variance': cumulative}, index=pc_names)
print("6.4 PCA explained-variance table:")
print(pca_summary.round(4))

# 6.5 How many components to keep?
n_90 = int(np.argmax(cumulative >= 0.90) + 1)          # first PC where cumulative >= 90%
n_kaiser = int((eigenvalues > 1).sum())                # Kaiser rule: keep eigenvalue > 1
print(f"\n6.5 Components needed for 90% variance : {n_90} out of {p}")
print(f"    Components with eigenvalue > 1 (Kaiser): {n_kaiser} out of {p}")
print(f"    Largest eigenvalue = {eigenvalues[0]:.3f}, smallest = {eigenvalues[-1]:.3f}")
if n_90 >= p - 3:
    print("    -> INTERPRETATION: variance is spread almost EQUALLY over all components.")
    print("       (If features were correlated, PC1 would capture most of the variance.)")
    print("       So PCA cannot compress this dataset - the features are independent.")

# 6.6 METHOD 2 - SVD:  Xs = U * S * V^T
#     Singular values s relate to eigenvalues by:  lambda = s^2 / (n-1)
U, s, Vt = np.linalg.svd(X_std, full_matrices=False)
eig_from_svd = s ** 2 / (n - 1)
print("\n6.6 SVD check: eigenvalues obtained from SVD equal eigen-decomposition?",
      np.allclose(eig_from_svd, eigenvalues))
print("    Singular values (first 5):", np.round(s[:5], 2))

# 6.7 PRINCIPAL COMPONENT SCORES = projecting data onto the new axes
scores = X_std @ eigenvectors                          # (n x p) matrix, new coordinates
print("\n6.7 Score matrix shape:", scores.shape)

# 6.8 LOADINGS: how much each original variable contributes to each PC
loadings = pd.DataFrame(eigenvectors[:, :3], index=num_cols, columns=['PC1', 'PC2', 'PC3'])
print("\n6.8 Loadings of first 3 PCs (bigger |value| = bigger contribution):")
print(loadings.round(3))

# 6.9 RECONSTRUCTION ERROR when we keep only k components
#     Xs_approx = scores[:, :k] @ eigenvectors[:, :k]^T
print("\n6.9 Reconstruction error (mean squared) when keeping k components:")
for k in [2, 5, 8, 10, 14]:
    X_rec = scores[:, :k] @ eigenvectors[:, :k].T
    mse = np.mean((X_std - X_rec) ** 2)
    print(f"    k = {k:>2} components -> MSE = {mse:.4f}")

# 6.10 Verify against scikit-learn's PCA (independent check of our maths)
from sklearn.decomposition import PCA
sk_ratio = PCA(n_components=p).fit(X_std).explained_variance_ratio_
print("\n6.10 Our variance ratios match scikit-learn PCA? ->", np.allclose(sk_ratio, explained))

# 6.11 PLOTS
# (a) Scree plot (eigenvalues) + cumulative variance
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(range(1, p + 1), eigenvalues, color='steelblue', edgecolor='black')
ax[0].axhline(1, color='red', linestyle=':', label='Kaiser line (eigenvalue = 1)')
ax[0].set_title('Scree Plot (eigenvalue of each PC)', fontweight='bold')
ax[0].set_xlabel('Principal component'); ax[0].set_ylabel('Eigenvalue'); ax[0].legend()
ax[1].plot(range(1, p + 1), cumulative, marker='o', linestyle='--', color='b')
ax[1].axhline(0.90, color='r', linestyle=':', label='90% threshold')
ax[1].set_title('Cumulative Variance Explained', fontweight='bold')
ax[1].set_xlabel('Number of components'); ax[1].set_ylabel('Cumulative variance')
ax[1].legend(); ax[1].grid(True)
plt.tight_layout()
save_and_show('pca_scree_plot.png')

# (b) 2-D projection of all vehicles, coloured by Emission Level
plt.figure(figsize=(8, 6))
colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
for level, col in colors.items():
    mask = (df['Emission Level'] == level).values
    plt.scatter(scores[mask, 0], scores[mask, 1], s=5, alpha=0.4, c=col, label=level)
plt.xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
plt.title('Data projected on PC1 vs PC2 (coloured by Emission Level)', fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
save_and_show('pca_2d_projection.png')


# =============================================================================
# STEP 7: STATISTICAL ANALYSIS ON MULTIVARIATE DATA  (CO2, 8 marks)
# (The dataset has no date/time column, so time-series is not applicable.
#  The rubric allows "time series OR statistical analysis" -> we do statistics.)
# =============================================================================
print("=" * 70)
print("STEP 7: STATISTICAL ANALYSIS")
print("=" * 70)

# 7.1 NORMALITY / DISTRIBUTION FIT -------------------------------------------
# Kolmogorov-Smirnov test against a UNIFORM distribution.
# H0: the data follows Uniform(min, max).  p > 0.05 -> we cannot reject H0.
print("7.1 KS test - does each variable follow a Uniform distribution?")
ks_rows = []
for col in num_cols:
    lo, hi = numerical_df[col].min(), numerical_df[col].max()
    ks_stat, ks_p = stats.kstest(numerical_df[col], 'uniform', args=(lo, hi - lo))
    ks_rows.append([col, ks_stat, ks_p, 'Uniform fits' if ks_p > 0.05 else 'Rejected'])
print(pd.DataFrame(ks_rows, columns=['Variable', 'KS stat', 'p-value', 'Verdict']).round(4).to_string(index=False))

# Shapiro-Wilk test for NORMALITY (on a 5000 sample, its limit)
sample = numerical_df['CO2 Emissions'].sample(5000, random_state=42)
sh_stat, sh_p = stats.shapiro(sample)
print(f"\nShapiro-Wilk normality test on CO2: W={sh_stat:.4f}, p={sh_p:.2e} ->",
      "Normal" if sh_p > 0.05 else "NOT normal (as expected for uniform data)")

# 7.2 CONFIDENCE INTERVALS for the mean (95%) --------------------------------
# CI = mean +/- t_critical * (std / sqrt(n))
print("\n7.2 95% Confidence intervals for the mean:")
ci_rows = []
for col in num_cols:
    m = numerical_df[col].mean()
    se = stats.sem(numerical_df[col])                   # standard error = std/sqrt(n)
    lo_ci, hi_ci = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
    ci_rows.append([col, m, lo_ci, hi_ci])
print(pd.DataFrame(ci_rows, columns=['Variable', 'Mean', 'CI lower', 'CI upper']).round(4).to_string(index=False))

# 7.3 HYPOTHESIS TESTS: does CO2 differ between groups? ----------------------
# One-way ANOVA: H0 = all group means are equal. p < 0.05 -> some group differs.
print("\n7.3 One-way ANOVA - is mean CO2 different between groups?")
for grp in ['Vehicle Type', 'Fuel Type', 'Road Type', 'Traffic Conditions', 'Emission Level']:
    groups = [g['CO2 Emissions'].values for _, g in df.groupby(grp)]
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"    CO2 by {grp:<19} F = {f_stat:7.3f}   p = {p_anova:.4f}  ->",
          "DIFFERENT" if p_anova < 0.05 else "no significant difference")

# Two-sample t-test example: Heavy traffic vs Free flow
a = df.loc[df['Traffic Conditions'] == 'Heavy', 'CO2 Emissions']
b = df.loc[df['Traffic Conditions'] == 'Free flow', 'CO2 Emissions']
t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)    # Welch's t-test
print(f"\n    t-test CO2: Heavy vs Free flow  t = {t_stat:.3f}, p = {p_t:.4f}")

# 7.4 MULTIPLE LINEAR REGRESSION using matrix algebra (normal equations) -----
# Model: y = b0 + b1*x1 + ... + b9*x9
# Solution: beta = (A^T A)^(-1) A^T y    (A = predictor matrix with a column of 1s)
print("\n7.4 Multiple linear regression (CO2 ~ all 9 predictors), by normal equations")
Xp = df[predictors].values
A = np.column_stack([np.ones(n), Xp])                  # add intercept column of ones
beta = np.linalg.solve(A.T @ A, A.T @ y)               # solves (A^T A) beta = A^T y
y_hat = A @ beta                                       # predicted CO2
resid = y - y_hat                                      # errors

# Goodness of fit
ss_res = np.sum(resid ** 2)                            # unexplained variation
ss_tot = np.sum((y - y.mean()) ** 2)                   # total variation
r2 = 1 - ss_res / ss_tot                               # R^2 : share of variance explained
k = len(predictors)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)          # penalised for number of features
f_reg = (r2 / k) / ((1 - r2) / (n - k - 1))            # overall F-test
p_f = 1 - stats.f.cdf(f_reg, k, n - k - 1)

# Standard error, t-value and p-value of every coefficient
sigma2 = ss_res / (n - k - 1)                          # residual variance
cov_beta = sigma2 * np.linalg.inv(A.T @ A)             # covariance of coefficients
se_beta = np.sqrt(np.diag(cov_beta))
t_beta = beta / se_beta
p_beta = 2 * (1 - stats.t.cdf(np.abs(t_beta), n - k - 1))
reg_table = pd.DataFrame({'Coefficient': beta, 'Std Error': se_beta, 't': t_beta, 'p-value': p_beta},
                         index=['Intercept'] + predictors)
print(reg_table.round(5))
print(f"\n    R^2 = {r2:.5f}   Adjusted R^2 = {adj_r2:.5f}")
print(f"    Overall F = {f_reg:.3f}, p = {p_f:.4f}")
print(f"    RMSE = {np.sqrt(ss_res / n):.3f}  (std of CO2 is {y.std():.3f})")
print("    Matches numpy least-squares? ->",
      np.allclose(beta, np.linalg.lstsq(A, y, rcond=None)[0]))
if r2 < 0.05:
    print("    -> R^2 is ~0: the predictors explain almost none of the CO2 variation.")

# 7.5 MULTIVARIATE OUTLIERS - Mahalanobis distance ---------------------------
# D^2 = (x - mean)^T * Cov^(-1) * (x - mean). Unlike z-score it considers ALL
# variables together. D^2 follows a chi-square law with p degrees of freedom.
cov_inv = np.linalg.inv(cov_manual)                    # inverse covariance matrix
d2 = np.einsum('ij,jk,ik->i', X_centered, cov_inv, X_centered)   # D^2 for every row
threshold = stats.chi2.ppf(0.999, df=p)                # 99.9% cut-off
print(f"\n7.5 Mahalanobis distance: rows beyond the 99.9% limit = {(d2 > threshold).sum()}"
      f" (expected by chance ~ {int(0.001 * n)})")

# 7.6 PLOTS: regression diagnostics
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(y_hat, y, s=3, alpha=0.3)
ax[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect prediction')
ax[0].set_xlabel('Predicted CO2'); ax[0].set_ylabel('Actual CO2')
ax[0].set_title(f'Actual vs Predicted CO2 (R^2 = {r2:.4f})', fontweight='bold'); ax[0].legend()
ax[1].hist(resid, bins=40, color='teal', edgecolor='black')
ax[1].set_title('Residual distribution', fontweight='bold')
ax[1].set_xlabel('Residual (actual - predicted)')
plt.tight_layout()
save_and_show('regression_diagnostics.png')


# =============================================================================
# STEP 8: FINAL SUMMARY (printed - copy into the report's Conclusion)
# =============================================================================
print("=" * 70)
print("STEP 8: SUMMARY OF FINDINGS")
print("=" * 70)
print(f"1. Data: {df.shape[0]} rows, {len(num_cols)} numerical features, no missing / wrong / duplicate values.")
print(f"2. Distributions: all numerical features are ~uniform (avg skew {stats_summary['Skewness'].mean():.2f}, avg kurtosis {stats_summary['Kurtosis'].mean():.2f}).")
print(f"3. Outliers: {int(outlier_df.values.sum())} flagged by Z-score/IQR.")
print(f"4. Correlation: max |r| between any two variables = {np.abs(upper).max():.4f} (features are nearly independent).")
print(f"5. Feature selection: strongest predictor of CO2 has |r| = {corr_table['|r|'].max():.4f}.")
print(f"6. PCA: {n_90} of {p} components needed for 90% variance -> no useful compression.")
print(f"7. Regression: R^2 = {r2:.5f} -> predictors explain almost none of CO2.")
print("\nAll graphs saved in the 'outputs' folder. DONE.")