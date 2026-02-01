import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(".1f")
    print(".1f")

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()

    # Fit on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data
    X_test_scaled = scaler.transform(X_test)

    print("Features scaled using StandardScaler")

    return X_train_scaled, X_test_scaled, scaler

def train_linear_regression(X_train, y_train):
    """
    Train Linear Regression model
    """
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    print("Linear Regression model trained successfully")
    print(f"Intercept: {model.intercept_:.2f}")
    print("Coefficients:")
    for feature, coef in zip(X_train.columns if hasattr(X_train, 'columns') else range(len(model.coef_)), model.coef_):
        print(f"  {feature}: {coef:.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using MSE and R² score
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    # Interpretation
    print("\nInterpretation:")
    print("- Lower MSE indicates better fit (lower prediction errors)")
    print(f"- Current MSE of {mse:.4f} means average squared prediction error")
    print("- R² closer to 1 indicates better fit (higher proportion of variance explained)")
    print(f"- Current R² of {r2:.4f} means {r2*100:.1f}% of variance is explained by the model")

    return y_pred, mse, r2

def plot_predictions(y_test, y_pred):
    """
    Plot actual vs predicted values
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()

def plot_residuals(y_test, y_pred):
    """
    Plot residuals to check model assumptions
    """
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

    # Residuals distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Residuals Distribution')
    plt.show()

def interpret_coefficients(model, feature_names):
    """
    Interpret the model coefficients
    """
    print("\nCoefficient Interpretation:")
    print("Each coefficient represents the change in target variable for a one-unit increase in the feature,")
    print("holding all other features constant.\n")

    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

    print(coefficients)

    return coefficients

def main(X, y):
    """
    Main modeling pipeline
    """
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train model
    model = train_linear_regression(X_train_scaled, y_train)

    # Evaluate model
    y_pred, mse, r2 = evaluate_model(model, X_test_scaled, y_test)

    # Visualizations
    plot_predictions(y_test, y_pred)
    plot_residuals(y_test, y_pred)

    # Interpret coefficients
    feature_names = X.columns.tolist()
    coefficients = interpret_coefficients(model, feature_names)

    return model, scaler, mse, r2, coefficients

if __name__ == "__main__":
    # For testing
    from data_cleaning import main as clean_data

    X, y, df = clean_data("../data/housing.csv")
    model, scaler, mse, r2, coefficients = main(X, y)
    print("Linear regression modeling completed successfully!")
