# 🏗️ Construction Cost Estimation with Machine Learning

This project applies data science techniques to solve a critical real-world problem in the construction industry: **accurately estimating total construction costs**. Using Python and machine learning models, I built a predictive pipeline that helps identify key cost drivers and forecast construction expenses using historical project data.

## 🧾 Background

Accurate construction cost estimation is crucial for:
- Managing budgets and resources efficiently
- Avoiding project overruns and financial risk
- Winning competitive bids in the construction industry

Traditional methods are often manual or based on rough heuristics. This project demonstrates how **data science and regression modeling** can improve transparency, accuracy, and efficiency in construction planning.

## 📊 Steps in the analysis : 

### 1. 📥 Data Collection & Cleaning
- Dataset loaded from CSV (`construction_estimates.csv`)
- Removed irrelevant features like `Policy_Reason`
- Verified data types and checked for nulls
- Summary statistics and initial EDA

### 2. 🔎 Exploratory Data Analysis (EDA)
- Correlation heatmaps revealed strong relationships between:
  - `Material_Cost`, `Labor_Cost`, and `Total_Estimate`
- Distribution checks for key numerical variables
- Insights into how different cost components influence total budget

### 3. 🤖 Modeling Approaches

#### ✅ Linear Regression
- Split data into training/testing sets (80/20)
- Fit model with `sklearn.linear_model.LinearRegression`
- Evaluation Metrics:
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  
#### 🧮 StatsModels (OLS + Quantile Regression)
- Used `statsmodels.api.OLS` for statistical insight into coefficients
- Performed quantile regression to estimate upper-bound costs (90th percentile)
- Useful for understanding worst-case or premium budget scenarios

#### 🧠 K-Nearest Neighbors (KNN) Regression
- Used `KNeighborsRegressor` with `n_neighbors=120`
- Compared performance to linear models
- Visualized predicted vs actual cost estimates

#### 🧪 PCA & Cross-Validation (exploratory section)
- Setup for PCA and RepeatedKFold CV for future model enhancements

## 📈 Key Results

- Linear Regression achieved a reasonably strong **R² Score**.
- KNN Regression provided visually tight predictions in scatterplots.
- Quantile regression identified high-end budget boundaries — useful for contingency planning.
- Correlation analysis confirmed that **Material and Labor costs** are the most significant drivers.

## Citations

 https://www.listendata.com/2018/01/linear-regression-in-python.html

 https://www.geeksforgeeks.org/k-nearest-neighbors-knn-regression-with-scikit-learn/

 https://www.statology.org/correlation-matrix-python/

 https://www.statology.org/principal-components-regression-in-python/

 https://www.statology.org/quantile-regression-in-python/
