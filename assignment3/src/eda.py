import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor # type: ignore

def univariate_analysis(df, numerical_cols, categorical_cols=None):
    """
    Perform univariate analysis on numerical and categorical variables
    """
    print("=== Univariate Analysis ===")

    # Numerical variables
    print("\nNumerical Variables:")
    for col in numerical_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std: {df[col].std():.2f}")
        print(f"  Min: {df[col].min():.2f}")
        print(f"  Max: {df[col].max():.2f}")
        print(f"  Skewness: {df[col].skew():.2f}")

        # Histogram
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    # Categorical variables (if any)
    if categorical_cols:
        print("\nCategorical Variables:")
        for col in categorical_cols:
            print(f"\n{col} value counts:")
            print(df[col].value_counts())

def bivariate_analysis(df, target_col, numerical_cols):
    """
    Perform bivariate analysis between features and target
    """
    print("\n=== Bivariate Analysis ===")

    # Scatter plots with target
    for col in numerical_cols:
        if col != target_col:
            plt.figure(figsize=(8, 4))
            sns.scatterplot(x=df[col], y=df[target_col])
            plt.title(f'{col} vs {target_col}')
            plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    print("\nCorrelation with target:")
    correlations = df[numerical_cols].corr()[target_col].sort_values(ascending=False)
    print(correlations)

def check_multicollinearity(df, numerical_cols, threshold=5.0):
    """
    Check for multicollinearity using VIF
    """
    print("\n=== Multicollinearity Check (VIF) ===")

    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numerical_cols
    vif_data["VIF"] = [variance_inflation_factor(df[numerical_cols].values, i)
                       for i in range(len(numerical_cols))]

    print(vif_data)

    # Identify high VIF features
    high_vif = vif_data[vif_data["VIF"] > threshold]
    if not high_vif.empty:
        print(f"\nFeatures with high VIF (> {threshold}):")
        print(high_vif)
    else:
        print(f"\nNo features with VIF > {threshold}")

    return vif_data

def plot_feature_importance(corr_with_target, top_n=10):
    """
    Plot feature importance based on correlation with target
    """
    plt.figure(figsize=(10, 6))
    top_features = corr_with_target.abs().nlargest(top_n)
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f'Top {top_n} Features by Correlation with Target')
    plt.xlabel('Absolute Correlation')
    plt.show()

def main(df, target_col='median_house_value'):
    """
    Main EDA function
    """
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # Univariate analysis
    univariate_analysis(df, numerical_cols, categorical_cols)

    # Bivariate analysis
    bivariate_analysis(df, target_col, numerical_cols)

    # Multicollinearity check
    vif_data = check_multicollinearity(df, numerical_cols)

    # Feature importance
    correlations = df[numerical_cols].corr()[target_col]
    plot_feature_importance(correlations)

    return vif_data, correlations

if __name__ == "__main__":
    # For testing
    from data_cleaning import load_data, handle_missing_values, remove_outliers, feature_engineering

    df = load_data("../data/housing.csv")
    df = handle_missing_values(df)
    numerical_cols = ['median_income', 'median_house_value', 'total_rooms', 'total_bedrooms']
    df = remove_outliers(df, numerical_cols)
    df = feature_engineering(df)

    vif_data, correlations = main(df)
    print("EDA completed successfully!")
