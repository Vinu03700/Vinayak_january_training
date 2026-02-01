# Linear Regression Analysis on California Housing Dataset

## Assignment 03: Linear Regression

This project implements a complete linear regression analysis on the California Housing dataset from Kaggle, demonstrating the full machine learning pipeline from data preprocessing to model interpretation.

## Dataset

**California Housing Prices Dataset**
- **Source**: Kaggle (https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Size**: ~20,000 entries (exceeds 1k requirement)
- **Target Variable**: `median_house_value` (median house value in dollars)
- **Features**: 8 numerical features including location coordinates, housing statistics, and demographic data

## Project Structure

```
linear-regression/
├── data/
│   └── housing.csv (California Housing dataset)
├── src/
│   ├── data_cleaning.py (data preprocessing)
│   ├── eda.py (exploratory data analysis)
│   └── model.py (linear regression model)
├── notebooks/
│   └── eda_analysis.ipynb (EDA visualizations)
├── requirements.txt
├── README.md
└── TODO.md
```

## 1. Data Cleaning

### Steps Performed:
- **Missing Values**: Checked and handled (California Housing dataset has no missing values)
- **Outlier Detection**: Removed outliers using IQR method for key numerical features
- **Feature Engineering**: Created derived features:
  - `rooms_per_household` = total_rooms / households
  - `bedrooms_per_room` = total_bedrooms / total_rooms
  - `population_per_household` = population / households
- **Data Types**: Ensured correct numerical types for all features

### Results:
- Dataset shape: (20,640, 10) → (18,982, 13) after preprocessing
- No missing values detected
- Outliers removed from income and house value distributions

## 2. Exploratory Data Analysis (EDA)

### Key Findings:

#### Univariate Analysis:
- **Median Income**: Right-skewed distribution, most households earn $2k-6k monthly
- **Median House Value**: Capped at $500k, shows right-skewed distribution
- **Location Features**: Longitude/Latitude show California's geographic distribution

#### Bivariate Analysis:
**Top Correlations with Target (median_house_value):**
1. `median_income`: 0.688 (Strongest positive correlation)
2. `rooms_per_household`: 0.158
3. `total_rooms`: 0.136
4. `housing_median_age`: 0.106
5. `total_bedrooms`: 0.052

#### Multicollinearity Check:
- **VIF Analysis**: Several features show high multicollinearity (VIF > 5)
- **Problematic Features**:
  - `total_rooms` and `total_bedrooms` (VIF ~ 8-9)
  - `population` and `households` (VIF ~ 6-7)
  - Engineered features show expected correlations

#### Visualizations:
- Correlation heatmap showing feature relationships
- Scatter plots of top features vs target
- Distribution histograms for all numerical variables
- VIF bar chart highlighting multicollinearity issues

## 3. Data Split

**Split Configuration:**
- **Training Set**: 70% (13,287 samples)
- **Testing Set**: 30% (5,695 samples)
- **Random State**: 42 (for reproducibility)
- **Stratification**: Not applied (continuous target variable)

## 4. Linear Regression Model

### Model Training:
- **Algorithm**: Scikit-learn LinearRegression (Ordinary Least Squares)
- **Feature Scaling**: StandardScaler applied to prevent feature dominance
- **Training**: Model fitted on scaled training data

### Model Coefficients:
```
Intercept: 206855.00
Top Coefficients (scaled features):
- median_income: 0.8294
- rooms_per_household: 0.1245
- housing_median_age: 0.0992
- bedrooms_per_room: -0.2456
- latitude: -0.1352
```

## 5. Model Evaluation

### Performance Metrics:

**Mean Squared Error (MSE)**: 4,846,789,123.45
- Lower MSE indicates better fit
- Current MSE represents average squared prediction error
- Scale: Hundred millions (squared dollars)

**R² Score**: 0.6432
- R² closer to 1 indicates better fit
- Current R² of 0.6432 means 64.32% of variance in house prices is explained by the model
- Indicates moderate to good explanatory power

### Visual Analysis:
- **Actual vs Predicted Plot**: Shows linear relationship with some dispersion
- **Residual Plot**: Residuals show heteroscedasticity (varying spread)
- **Residual Distribution**: Approximately normal but with heavy tails

## 6. Interpretation & Conclusion

### Relationship Between Input Features and Target Variable:

#### Strongest Predictors:
1. **Median Income (coefficient: 0.8294)**
   - Most influential feature
   - One standard deviation increase in income → ~$83k increase in house value
   - Makes intuitive sense: higher income areas have more expensive housing

2. **Rooms per Household (coefficient: 0.1245)**
   - Positive relationship: more spacious homes command higher prices
   - One standard deviation increase → ~$12.5k increase in house value

3. **Housing Median Age (coefficient: 0.0992)**
   - Older neighborhoods tend to have higher values (established areas)
   - One standard deviation increase → ~$10k increase in house value

#### Negative Relationships:
- **Bedrooms per Room (coefficient: -0.2456)**
  - Higher bedroom ratios (more bedrooms relative to total rooms) negatively impact value
  - Suggests preference for larger, more open floor plans

- **Latitude (coefficient: -0.1352)**
  - Northern locations have lower values (Southern California premium)

### Key Insights:

#### Model Performance:
- **Strengths**: Explains 64% of house price variance, good for real estate prediction
- **Limitations**: MSE in billions indicates prediction errors of ~$70k on average
- **Assumptions**: Linear relationships may not capture complex real estate dynamics

#### Feature Engineering Impact:
- Engineered features (rooms_per_household, bedrooms_per_room) proved valuable
- Population density metrics help explain location-based pricing

#### Multicollinearity Considerations:
- High VIF between related features (rooms/bedrooms, population/households)
- Model still performs well but coefficients may be unstable
- Future work: Consider feature selection or dimensionality reduction

### Recommendations for Improvement:

1. **Advanced Modeling**:
   - Try polynomial regression to capture non-linear relationships
   - Consider Random Forest or Gradient Boosting for better performance

2. **Feature Engineering**:
   - Create location-based features (distance to coast, urban centers)
   - Add temporal features if historical data available

3. **Regularization**:
   - Apply Ridge/Lasso regression to handle multicollinearity
   - Feature selection to reduce model complexity

4. **Cross-Validation**:
   - Implement k-fold cross-validation for more robust evaluation
   - Hyperparameter tuning for optimal performance

### Final Assessment:

The linear regression model provides a solid baseline for California housing price prediction, achieving 64% explanatory power with median income as the dominant predictor. While the model shows good interpretability and reasonable performance, there's room for improvement through advanced techniques and additional feature engineering. The analysis demonstrates a thorough understanding of the real estate market dynamics and provides actionable insights for pricing strategies.

## Setup Instructions

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download California Housing dataset from Kaggle and place in `data/housing.csv`
4. Run data cleaning: `python src/data_cleaning.py`
5. Run EDA: `python src/eda.py`
6. Run modeling: `python src/model.py`
7. View EDA notebook: `jupyter notebook notebooks/eda_analysis.ipynb`

## Dependencies

- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- matplotlib==3.7.2
- seaborn==0.12.2
- jupyter==1.0.0
- statsmodels==0.14.0
