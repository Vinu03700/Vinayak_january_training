# Test04 - Supervised Machine Learning Models

## 📌 Project Title

Prediction of Titanic Passenger Survival Using Supervised Machine Learning

---

## 📖 Problem Statement

Build prediction models to determine whether a passenger survived the Titanic disaster using multiple supervised machine learning algorithms.

---

## 📊 Dataset

Titanic Dataset (Kaggle)

Rows: 891  
Target Variable: Survived  

Features Used:

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

---

## 🧹 Data Preprocessing

1. Removed duplicate records  
2. Filled missing Age values using mean  
3. Filled missing Embarked values using mode  
4. Removed irrelevant columns (PassengerId, Name, Ticket, Cabin)  
5. Encoded categorical variables using Label Encoding  
6. Applied feature scaling using StandardScaler  
7. Split dataset into 80% training and 20% testing  

---

## 🤖 Algorithms Used

1. Linear Regression  
2. Decision Tree  
3. Random Forest  
4. K-Nearest Neighbors (KNN)  
5. Support Vector Machine (SVM)

---

## 📈 Evaluation Metrics

Classification Metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  

---

## 📊 Sample Results

Linear Regression Accuracy: ~78%  
Decision Tree Accuracy: ~80%  
Random Forest Accuracy: ~84%  
KNN Accuracy: ~82%  
SVM Accuracy: ~83%  

---

## ✅ Conclusion

Random Forest and SVM produced the highest accuracy.  
Data preprocessing significantly improved model performance.  
Multiple algorithms were successfully implemented and compared.

---

## ⚙️ Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  
Seaborn  

---

## ▶️ How To Run

pip install -r requirements.txt  
python ml_models.py
