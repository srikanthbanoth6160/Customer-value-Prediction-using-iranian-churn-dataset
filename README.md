# 📊 Customer Value Prediction using Iranian Churn Dataset  

## 🚀 Introduction  
This project implements statistical and machine learning methods to predict **customer value** using the **Iranian Churn dataset**. Customer value is a measure of a customer's importance to a business, and predicting it can help optimize **customer retention strategies** and **business decisions**.  

## 🎯 Objective  
The primary goal is to predict **customer value** using 12 predictor variables from an Iranian telecom dataset. The project aims to:  
- Identify key factors influencing customer value.  
- Predict customer churn based on statistical learning techniques.  
- Optimize business decisions for better customer retention.  

## 📂 Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/)  
- **Instances:** 3,150 customers  
- **Features:** 14 variables, including:  

  **Response Variable:**  
  - **Customer Value** (target variable)  

  **Predictor Variables:**  
  - Call Failures  
  - Complaints  
  - Subscription Length  
  - Charge Amount  
  - Seconds of Use  
  - Frequency of Use  
  - Frequency of SMS  
  - Distinct Called Numbers  
  - Age Group  
  - Tariff Plan  
  - Status  
  - Age  
  - Churn  

### 📊 Data Insights  
- Most customers have **few call failures**.  
- **75% of customers** reported no complaints.  
- Subscription length varies from **3 to 47 months**.  
- The median customer age is **30 years**.  
- **15.71% of customers have churned**.  

## 🔍 Methodology  

### 📈 Exploratory Data Analysis (EDA)  
- Histograms, box plots, and scatter plots were used to visualize relationships.  
- **Correlations** were identified between variables (e.g., **Age** and **Age Group**).  

### 📊 Model Evaluation  
- Models were assessed using **Mean Squared Error (MSE)**, **Adjusted R-squared**, **AIC**, **BIC**, and **Cp**.  

## 🏆 Results  

### 🔹 Linear Models  
| Model                  | Adjusted R² | Test MSE  |  
|------------------------|-------------|------------|  
| **Linear Regression**  | 98.2%       | -          |  
| **Regression Subsets** | -           | -          |  

- **Best Subset Model** identified 9 key predictors.  
- Forward and Backward Selection were used for feature selection.  

### 🔹 Regularization Models  
| Model        | Test MSE  |  
|-------------|----------|  
| **Ridge Regression**  | 6562.899  |  
| **Lasso Regression**  | 6563.246  |  

- **Lasso Regression** helped in feature selection by shrinking some coefficients to zero.  

### 🔹 Non-Linear Models  
| Model                      | Test MSE  |  
|----------------------------|----------|  
| **Polynomial Regression**  | -        |  
| **Generalized Additive Models (GAM)** | **3409.622**  |  

- **GAM outperformed polynomial regression**, capturing **non-linear relationships** effectively.  

### 🔹 Tree-Based Models  
| Model              | Test MSE  |  
|--------------------|----------|  
| **Regression Tree** | 14181.29 |  
| **Random Forest**  | **1266.552**  |  

- **Random Forest significantly improved accuracy**, reducing test MSE.  

### 🔹 Support Vector Machine (SVM)  
- **Radial Kernel SVM achieved a test MSE of 1014.764**.  

### 🔹 Neural Networks  
| Model              | Test MSE  |  
|--------------------|----------|  
| **Best Neural Network Model** | **469.2368** |  

- **Neural networks outperformed all models**, capturing complex patterns in the dataset.  

## 📍 Challenges  
- Handling **high dimensionality** and **feature selection**.  
- Identifying the **best regularization techniques**.  
- Balancing **model complexity** with performance.  

## 🚀 Future Improvements  
- **Hyperparameter tuning** to further optimize model performance.  
- **Deep learning techniques** for improved feature extraction.  
- **Deployment** as a real-time predictive tool for business decision-making.  

## ⚙️ Dependencies  
- `ggplot2`  
- `leaps`  
- `glmnet`  
- `randomForest`  
- `neuralnet`  
- `pls`  

## 📖 References  
- Awad, M. K. (2015). *Support Vector Regression.*  
- Iranian Churn Dataset. (n.d.). Retrieved from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/).  
- James, G. W. (2023). *Support Vector Machines.*  

## 👨‍💻 Author  
- **Srikanth Banoth**  
