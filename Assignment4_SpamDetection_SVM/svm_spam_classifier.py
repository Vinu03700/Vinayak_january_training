# Assignment 04 - Spam Email Detection using SVM

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------

data = pd.read_csv("dataset/spam.csv", encoding='latin-1')
data = data[['v1','v2']]
data.columns = ['label','message']

print(data.head())
print("Dataset Size:", data.shape)

# -----------------------------
# 2. Data Cleaning
# -----------------------------

# Convert labels to numbers
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Check missing values
print(data.isnull().sum())

# -----------------------------
# 3. Feature Extraction (Text → Numbers)
# -----------------------------

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']

# -----------------------------
# 4. Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Build SVM Model
# -----------------------------

model = SVC(kernel='linear')
model.fit(X_train, y_train)

# -----------------------------
# 6. Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# 7. Model Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 8. Test Custom Message
# -----------------------------

sample = ["Congratulations! You won a free lottery ticket"]
sample_vec = vectorizer.transform(sample)
result = model.predict(sample_vec)

if result[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")
