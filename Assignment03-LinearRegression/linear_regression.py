# Assignment 03 - Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# 1. Load Dataset
# ----------------------------
data = pd.read_csv("dataset/train.csv")
print("Dataset Shape:", data.shape)
print(data.head())

# ----------------------------
# 2. Data Cleaning
# ----------------------------

# Selecting important features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual']
target = 'SalePrice'

df = data[features + [target]]

# Check missing values
print(df.isnull().sum())

# Drop missing values
df = df.dropna()

# ----------------------------
# 3. Exploratory Data Analysis
# ----------------------------

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Scatter plot
sns.scatterplot(x="GrLivArea", y="SalePrice", data=df)
plt.title("Living Area vs Sale Price")
plt.show()

# ----------------------------
# 4. Data Split
# ----------------------------

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 5. Build Linear Regression Model
# ----------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# 6. Predictions
# ----------------------------

y_pred = model.predict(X_test)

# ----------------------------
# 7. Model Evaluation
# ----------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# ----------------------------
# 8. Coefficients
# ----------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coefficients)
