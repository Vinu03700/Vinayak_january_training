# Test04 - Supervised Machine Learning Models

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# -----------------------------------
# 1. Load Dataset
# -----------------------------------

data = pd.read_csv("dataset/titanic.csv")
print(data.head())

# -----------------------------------
# 2. Data Cleaning & Preprocessing
# -----------------------------------

# Remove duplicates
data.drop_duplicates(inplace=True)

# Fill missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop irrelevant columns
data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])

# -----------------------------------
# 3. Feature & Target
# -----------------------------------

X = data.drop('Survived', axis=1)
y = data['Survived']

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------------
# 4. Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# 5. Models
# -----------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

# -----------------------------------
# 6. Training & Evaluation
# -----------------------------------

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if name == "Linear Regression":
        predictions = np.round(predictions)

    print("\n---------------------------")
    print(name)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
