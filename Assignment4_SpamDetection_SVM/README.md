# Assignment 04 – Spam Email Detection using SVM

## 📌 Objective
Build a Support Vector Machine (SVM) model to detect spam emails and evaluate its performance.

---

## 📊 Dataset
Dataset: Spam Email Dataset (SMS Spam Collection)  
Source: Kaggle  
Total Records: 5572  

Target Variable:
label  
- 0 → Ham (Not Spam)  
- 1 → Spam  

Feature:
message (email text)

---

## 🧹 1. Data Cleaning
- Selected required columns.
- Renamed columns.
- Converted labels to numeric values.
- Checked for missing values.

---

## 🔡 2. Feature Engineering
- Used TF-IDF Vectorizer.
- Converted text messages into numerical vectors.

---

## 🔀 3. Data Splitting
- Training Data: 80%
- Testing Data: 20%

---

## 🤖 4. Model Building
Algorithm: Support Vector Machine (SVM)  
Kernel: Linear  

Model trained using training data.

---

## 📉 5. Model Evaluation

Metrics Used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Example Output:
Accuracy ≈ 0.97 (97%)

High accuracy indicates strong spam detection performance.

---

## 🔍 6. Prediction
Model can classify new messages as:
- Spam
- Not Spam

Example:
"Congratulations! You won a free lottery ticket" → Spam

---

## ✅ Conclusion
The SVM model performs very well for spam detection.  
TF-IDF with Linear SVM is effective for text classification problems.  
The system can be used as a basic email spam filter.

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## ▶️ How to Run

pip install -r requirements.txt  
python svm_spam_classifier.py
