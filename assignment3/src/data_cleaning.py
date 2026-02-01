import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load the California Housing dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def explore_data(df):
    """
    Basic exploration of the dataset
    """
    print("Dataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values before handling:\n{missing_values}")

    # For California Housing dataset, there are typically no missing values
    # But if there were, we would handle them here
    # For demonstration, let's assume we need to handle some missing values

    # Fill missing values with median for numerical columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    print(f"Missing values after handling:\n{df.isnull().sum()}")
    return df

def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Remove outliers using IQR method
    """
    df_clean = df.copy()

    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Remove outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    print(f"Dataset shape after outlier removal: {df_clean.shape}")
    return df_clean

def feature_engineering(df):
    """
    Create new features if needed
    """
    # Create rooms per household
    df['rooms_per_household'] = df['total_rooms'] / df['households']

    # Create bedrooms per room
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']

    # Create population per household
    df['population_per_household'] = df['population'] / df['households']

    print("New features created:")
    print("- rooms_per_household")
    print("- bedrooms_per_room")
    print("- population_per_household")

    return df

def prepare_features_target(df, target_column='median_house_value'):
    """
    Separate features and target variable
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y

def main(filepath):
    """
    Main data cleaning pipeline
    """
    # Load data
    df = load_data(filepath)

    # Explore data
    explore_data(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Remove outliers from key numerical columns
    numerical_cols = ['median_income', 'median_house_value', 'total_rooms', 'total_bedrooms']
    df = remove_outliers(df, numerical_cols)

    # Feature engineering
    df = feature_engineering(df)

    # Prepare features and target
    X, y = prepare_features_target(df)

    return X, y, df

if __name__ == "__main__":
    # For testing
    X, y, df = main("../data/housing.csv")
    print("Data cleaning completed successfully!")
