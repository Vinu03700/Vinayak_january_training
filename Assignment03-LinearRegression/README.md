# Assignment 03 – Linear Regression

## 📌 Objective
Build a Linear Regression model on a real-world dataset, analyze relationships between variables, and evaluate model performance.

---

## 📊 Dataset
Dataset: House Prices – Advanced Regression Techniques  
Source: Kaggle  
Total Records: 1460  

Target Variable:
SalePrice (House Sale Price)

Selected Features:
- GrLivArea (Above ground living area)
- BedroomAbvGr (Number of bedrooms)
- FullBath (Number of bathrooms)
- OverallQual (Overall material quality)

---

## 🧹 1. Data Cleaning
- Selected relevant numerical columns.
- Checked for missing values.
- Removed rows with missing values.

---

## 📈 2. Exploratory Data Analysis (EDA)
- Correlation heatmap used to check relationships.
- Scatter plot between living area and sale price.
- Strong positive correlation observed between:
  - GrLivArea and SalePrice
  - OverallQual and SalePrice

Multicollinearity:
- No extremely high correlations between independent variables.

---

## 🔀 3. Data Splitting
- Training Set: 80%
- Testing Set: 20%
- Used train_test_split() from scikit-learn.

---

## 🤖 4. Model Building
- Algorithm: Linear Regression
- Library: scikit-learn
- Model trained using training data.

---

## 📉 5. Model Evaluation

Metrics Used:
- Mean Squared Error (MSE)
- R² Score

Example Output:
Mean Squared Error: ~2,000,000,000  
R² Score: ~0.75  

Interpretation:
- Lower MSE indicates lower prediction error.
- R² close to 1 means good model fit.

---

## 📌 6. Feature Interpretation

Coefficients explain impact on SalePrice:

- GrLivArea: Positive → Larger area increases price
- BedroomAbvGr: Slight positive effect
- FullBath: Positive effect
- OverallQual: Strong positive effect

Higher coefficient = stronger influence.

---

## ✅ Conclusion
The Linear Regression model successfully predicts house prices using selected features.  
Living area and overall quality are the most important predictors.  
Model performance is reasonably good with R² around 0.75.

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## ▶️ How to Run

pip install -r requirements.txt  
python linear_regression.py
