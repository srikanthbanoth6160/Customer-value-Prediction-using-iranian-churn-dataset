# Customer Value Prediction using Iranian Churn Dataset

## Introduction
In this project, we implement statistical learning methods to address a real-world regression problem: customer value prediction using the Iranian Churn dataset. The goal is to predict the customer value for a given customer based on 12 variables. Customer value is a measure of how important a customer is to a company, and predicting this can help businesses maintain long-term relationships with customers and enhance service quality.

## Objective
The primary objective is to predict the customer value based on the available dataset, which is collected from an Iranian telecom company. The project aims to understand factors influencing customer value, predict customer churn, and optimize business decisions accordingly.

## Dataset
The dataset used in this project is the Iranian Churn Dataset, which is available at the UC Irvine Machine Learning Repository. It contains 3,150 observations (each corresponding to a customer) and 14 variables:

**Response variable:**
- Customer Value

**Predictor variables:**
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

### Data Insights:
- Most customers have few call failures.
- Around 75% have no complaints.
- The subscription length varies from 3 to 47 months.
- The charges are predominantly low (mean: 0.94).
- The median age of customers is 30.
- Approximately 15.71% of customers have churned.

## Methodology

### Descriptive Statistics:
We performed a statistical summary of the dataset to understand the distribution of each variable. We identified high correlations between some variables (e.g., Age and Age Group), which may allow for dimensionality reduction in future modeling.

### Exploratory Data Analysis:
We visualized the relationship between different predictors and customer value through histograms, box plots, and scatter plots. The relationship between customer value and age group was also analyzed.

### Model Evaluation:
The models were evaluated using Mean Squared Error (MSE), Adjusted R-squared, and other selection criteria like AIC, BIC, and Cp.

## Proposed Solutions

### Linear Models:
A linear regression model achieved an impressive Adjusted R-squared of 98.2%, identifying key predictors like subscription length, service usage time, and frequency of SMS.

### Regression Subsets:
We explored different combinations of predictors using Adjusted R-squared, BIC, and Cp for subset selection. The best-performing subset of 9 predictors was identified.

### Forward and Backward Selection:
Two feature selection methods were used—Forward Selection and Backward Selection—to build more efficient models.

### Ridge Regression:
Ridge regression was applied with cross-validation to select an optimal regularization strength. The test set MSE was 6562.899.

### Lasso Regression:
Lasso regression applied L1 regularization and achieved a test set MSE of 6563.246, assisting in feature selection.

### Non-linear Models:
We explored Polynomial Regression and Generalized Additive Models (GAM) to capture non-linear relationships in the data. GAM outperformed polynomial regression with a test MSE of 3409.622.

### Tree-based Models:
- **Regression Tree**: The decision tree model showed poor performance with a test MSE of 14181.29.
- **Random Forest**: The Random Forest model significantly improved performance with a test MSE of 1266.552.

### Support Vector Machine (SVM):
SVM was applied to predict customer value with a radial kernel, achieving a test MSE of 1014.764.

### Neural Networks:
Neural networks were explored with different architectures, and the best model achieved a test MSE of 469.2368.

## Discussion
In comparing all models, neural networks achieved the lowest test MSE, outperforming other methods like Random Forest, SVM, and GAM. The results indicate that more flexible models, such as polynomial regression, GAM, and neural networks, capture the underlying non-linearity of customer value more effectively than simpler models.

## Conclusion
We successfully built a predictive model for customer value using various statistical and machine learning techniques. While neural networks performed the best, Random Forest and SVM models also provided strong performance, suggesting that the relationship between predictors and customer value is complex and non-linear.

## Dependencies
- `ggplot2`
- `leaps`
- `glmnet`
- `randomForest`
- `neuralnet`
- `pls`

## References
- Awad, M. K. (2015). Support vector regression. In *Efficient learning machines: Theories, concepts, and applications for engineers and system designers*.
- Iranian Churn Dataset. (n.d.). Retrieved from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/).
- James, G. W. (2023). Support Vector Machines. In *An introduction to statistical learning*.
