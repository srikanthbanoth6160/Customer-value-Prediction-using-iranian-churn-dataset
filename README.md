Introduction
Objective
In this project, we implement statistical learning methods to address a real-world regression problem: customer value prediction using Iranian churn dataset. 
The goal of this project is to predict customer value based on 12 variables corresponding to a customer. The customer value, represented by a real number, is a measure of the importance of a customer to a company. We aim to build a model that predicts this value for a given customer using a data set collected from an Iranian telecom company. 
Prediction of customer value is crucial to successful business since it helps the company understand and predict customers’ behavior and provide desired services in order to build long-term relationships and maintain a competitive edge in the market. 
Data Set
The data set we use is Iranian Churn Dataset downloaded from UC Irvine Machine Learning Repository (Iranian Churn Dataset, n.d.). It contains 3150 observations, each corresponding to a customer, and 14 variables including the response variable Customer Value and 13 predictors: Call Failures, Complains, Subscription Length, Charge Amount, Seconds of Use, Frequency of Use, Frequency of SMS, Distinct Called Numbers, Age Group, Tariff Plan, Status, Age and Churn. 
This dataset helps an Iranian telecom company predict customer value and understand factors influencing it, including predicting customer churn. With 3150 entries and 14 variables like call problems and complaints, the goal is to improve services and reduce customer loss.
Building a regression model helps an Iranian telecom company predict customer value, anticipate churn, and focus on factors like call problems and complaints to enhance services and customer satisfaction, ultimately making informed business decisions.
This dataset can be found: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
Data Set Summary
In this telecom dataset:
•	Most customers have few call failures.
•	About 75% have no complaints.
•	Subscription length varies (3 to 47 months).
•	Charges are mostly low (mean 0.94).
•	Median usage is 2990 seconds.
•	Customers use services moderately.
•	Majority are in age groups 2 and 3.
•	Most have tariff plan 1 and status 1 (likely active).
•	Median age is 30, and mean customer value is 471.0.
•	Approximately 15.71% have churned, suggesting areas for improvement.

Methodology
Descriptive Statistics
We took a statistical summary of the data set to get a general idea of each variable. As all the qualitative features are already encoded as integers, we have only numerical variables.
 
Also, we created scatter plots for each pair of variables to find any correlation between variables. We identified two pairs highly correlated: Age and Age Group; Seconds of Use and Frequency of Use. Below are scatter plots of these pairs, from which we can see a clear linear trend indicating high correlation. The correlation values are 0.96, 0.95 respectively. We can consider using only one of each pair when selecting variables for our model. 
 

The data set has no missing values, so we do not need to clean the data. 
Exploratory Analysis
•	The grid of plots displays the distribution of customer values, variations by age group and tariff plan, and the relationship between seconds of use and customer values. It provides a visual overview of key aspects of the dataset.
 
•	Initially, we checked the size of the data set it has 3150 rows and 14 columns ,where Customer Value is the response variable and 13 predictor variables.
•	We split the data into training (2150 observations) and test sets (1000 observations). 

Evaluation Methods
For assessing the performance of proposed solution, we may use one or more of the following evaluation methods or criteria: 
•	Test Mean Squared Error (MSE): We hold out some of observations as a test set and compute MSE on the test set to compare performances of different models. So, we sampled 2150 observations as training set to fit models and evaluated the fitted models on the rest 1000 observations as test set. 
•	Adjusted R-squared, BIC, AIC or Cp to compare models with different numbers of variables in regression subset selection methods. 

Proposed Solutions
Linear Models
The linear model, with an impressive, adjusted R-squared of 98.2%, reveals key factors influencing customer value. Subscription length, service usage time, and frequency of SMS positively impact customer value, while higher charge amounts have a negative association. Tariff Plan 2 users tend to have higher customer value. The model's predictions closely align with actual values, contributing valuable insights for the telecom company.
 
Regression Subsets
Regression Subsets Analysis:
1.	Variable Selection:
•	Explored different combinations of 13 predictor variables to find the best subset.
2.	Metrics Used:
•	Used metrics like Adjusted R-squared, BIC, and RSS to evaluate subset performance.
3.	Best Subset (9 Variables):
•	Identified a top-performing subset with 9 key predictors, including Subscription Length, Charge Amount, Seconds of Use, and more.
4.	Subset Plots:
•	Plots showed changes in model metrics with varying numbers of variables. Optimal subset had 9 variables according to BIC.
 
5.	Key Predictors:
•	Found Subscription Length, Charge Amount, and others to be crucial predictors for customer value.
 
6.	Model Efficiency:
•	Resulting model is concise yet effective, tailoring predictions to telecom customer behavior.
Forward and Backward Selection
In building our predictive model for customer value, we employed two key methods—Forward Selection and Backward Selection. Forward Selection added predictors progressively, refining the model at each step, while Backward Selection started with a full set and iteratively removed less impactful predictors. We assessed model performance using metrics like Adjusted R-squared and Cp, ensuring a balance between model complexity and accuracy. 
 










Plots visualized the impact of variable additions/removals. Ultimately, these methods helped us pinpoint a subset of predictors for an efficient and effective model tailored to predict customer value in our telecom dataset.
Ridge Regression
In Ridge Regression, we explored various regularization strengths (lambda values) using cross-validation to find the optimal balance between model complexity and predictive performance. The selected lambda value was used to train the model, and the mean squared error on the test set was found to be 6562.899. Ridge Regression helps mitigate overfitting by penalizing large coefficients, contributing to a more robust and generalized predictive model for customer value in our telecom dataset.
 
 
Lasso
In Lasso Regression, we applied the L1 regularization technique to encourage sparsity in the model, effectively selecting a subset of the most influential predictor variables. After performing cross-validation to determine the optimal lambda (regularization strength), the selected model achieved a mean squared error of 6563.246 on the test set. Lasso Regression proves useful for feature selection and simplifying the model structure, contributing to an interpretable and potentially more efficient model for predicting customer value in our telecom dataset.
 
 
Non-linear Models
In order to see if there is any nonlinearity in the response-predictor relationship, we plot residuals vs. fitted values for the full linear model fitted with all the predictors. From the residual plot below, we can see some non-random patterns indicating that there may be nonlinear relationships between the response and some predictors. 
To address this nonlinear relationship, we developed two types of nonlinear models: polynomial regression models and generalized additive models (GAM). 
 
Polynomial Regression
To fit a polynomial regression model, we added quadratic terms to the full linear model, and compared different combinations of terms as in the table below. The best model has 20 variables (including linear and quadratic terms) with test MSE of 5559.735.
Number of terms	Test MSE
13	6562.927
14	6519.835
15	6442.024
16	6360.354
17	6194.252
18	6073.331
19	5664.355
20	5559.735 (lowest)
21	5586.074

The above table shows a decreasing trend in test MSE as the number of terms increases, until there are more than 20 terms when the test MSE starts to increase. This means that adding these quadratic terms up to 20 variables improves the model, and the more variables added, the better. 
Note: Quadratic terms of variables ‘Complains’, ‘Tariff Plan’, ‘Status’, ‘Age’ and ‘Churn’ are not considered because adding their quadratic terms results in NA-valued coefficients due to high linear dependency with existing variables. For example, the variable ‘Complains’ is a 0-1 binary variable (see the summary above) and taking the square of it produces itself as a new column which is highly correlated with the original column, which makes the model unable to estimate all coefficients. 
Details of the best polynomial model are as follows. 
 
Coefficients of the best polynomial model: 
 

GAM (Generalized Additive Model)
Another non-linear model to consider is generalized additive model (GAM). We fit GAMs with smoothing splines of different degrees of freedom. The following table shows that the best GAM model has 25 degrees of freedom with test MSE of 3409.622. 
Degree of freedom	Test MSE
5	4271.81
10	3675.925
15	3496.167
20	3411.413
25	3409.622 (lowest)
30	3449.193

Below are details of the best GAM model. 
 
For this data set, most nonlinear models perform better than linear models due to their high flexibility to address nonlinear relationships. Among nonlinear models, GAM has significantly better performance than polynomial regression since it is more flexible. We can see from the above two tables that even the worst GAM model outperforms the best polynomial model. 
Tree-based Methods
Regression Tree
In the Regression Tree model, the decision tree structure was employed to capture complex relationships within the data. However, the model resulted in a higher mean squared error of 14181.29 on the test set, indicating limitations in capturing the underlying patterns in predicting customer value. The visualization of the tree structure, provided by the rpart.plot package, offers insights into the decision-making process of the model.
 
 
Random Forest
In the Random Forest model, an ensemble of trees was employed to collectively predict customer value. The model achieved a relatively low mean squared error of 1266.552 on the test set, indicating its effectiveness in capturing patterns within the data.
 
 
The varImpPlot provides insights into the importance of different predictors, highlighting key factors influencing customer value. The ensemble approach of Random Forest contributes to its robustness and ability to handle complex relationships in the telecom dataset.

Support Vector Machine
Support vector machine (SVM) is an effective machine learning algorithm for both classification and regression tasks. For classification, SVM aims to find a hyperplane (decision boundary) that optimally separates different classes by maximizing the margin (distance to the closest observation) (James, 2023). 
For regression, the goal of SVM is to find a hyperplane that captures the trend of a continuous response. Specifically, the regression problem is formulated as an optimization problem that attempts to find the narrowest tube centered around the hyperplane, while minimizing the prediction error. That is, the tube is to contain as many observations as possible while satisfying the error constraint (Awad, 2015). 
In this project, we employ SVM regression to predict customer value. 
Support Vector Machine (SVM) with radial kernel and epsilon-regression type was applied to predict customer value.
 
The model demonstrated a mean squared error of 1014.764 on the test set, indicating its capability to capture underlying patterns in the data. SVM, with its flexibility in handling complex relationships, proves effective in predicting customer value in the telecom dataset.
 

Neural Networks
Neural Network is a popular advanced learning method which is inspired by the functions of neurons of human brain. A neural network is constructed by organizing nodes or units in layers, which includes an input layer, hidden layer(s) and an output layer. Each layer has its own set of parameters and activation functions to determine its output which is taken as input to the next layer. 
In this project, we explored different structures of neural networks, varying number of hidden layers and number of units in each layer. Each layer is defined with a ReLU activation function, and has a dropout rate of 0.1. The model is trained using the root mean square propagation (RMSProp) optimizer for 30 epochs with a batch size of 4. The results are shown below. 
Structure	Test MSE
Single layer, 64 units	4517.755
Single layer, 128 units	3569.743 (lowest single-layer)
Single layer, 256 units	3722.997
2 layers, 64-32 units	2673.896
2 layers, 64-64 units	2166.937
2 layers, 128-64 units	1441.178
2 layers, 128-128 units	1194.482
2 layers, 256-128 units	1159.809
2 layers, 256-256 units	929.9018
2 layers, 512-256 units	893.6971
2 layers, 512-512 units	469.2368 (lowest 2-layer)
3 layers, 64-64-64 units	1534.054
3 layers, 128-128-128 units	950.6605
3 layers, 256-256-256 units	874.404 (lowest 3-layer)
3 layers, 512-512-512 units	1191.603
3 layers, 512-256-256 units	1471.821
3 layers, 256-256-128 units	1040.973

We stopped increasing the number of units at 512 because the model training became significantly slow, spending about 5 seconds per epoch. 
The best NN model has 2 layers with 512 units per layer, the test MSE is 469.2368. 
From the table above, we see that for single layer model, the best number of units is 128, too few or too many units will degrade performance. For 2-layer models, there is a trend of improvement as the number of units increases, in either layer. For 3-layer structures, on the other hand, we found an optimal unit arrangement (256 units each layer) instead of a monotonic trend. 
In general, we expect a neural net with a deeper structure and more units to perform better. But it may not be the case when the structure becomes more complicated, as the redundancy of units and layers may produce redundant information resulting in suboptimal prediction power. In addition, computation time can be another disadvantage of deep and large networks having a huge number of parameters to estimate. 

Discussion
Now, we compare the results of different types of learning methods. The best test MSE achieved is selected from each type of learning method, listed in a table and plotted in a bar chart below.
It can be seen from the bar chart that flexible parametric methods such as polynomial, GAM, neural network outperform inflexible parametric methods like Ridge and Lasso, achieving higher performance with more flexibility, as GAM outperforms polynomial and neural network outperforms GAM. This indicates that the true relationship between the response and predictors is highly nonlinear. 
Non-parametric methods like random forest and SVM observe better performance than most parametric methods (Ridge, Lasso, Polynomial, GAM), which can be attributed to the high flexibility inherent in non-parametric methods that better captures nonlinear relationships. Although the regression tree performs poorly compared to all other methods due to overfitting, its drawback can be completely overcome by assembling into a random forest by enhancing diversity and robustness. 
The neural network achieved the lowest test MSE among all types of methods explored in this project, as a result of careful layer building. The power of neural network is given by its specific structure and a large set of parameters to account for high flexibility.  A drawback is that it may come at a price of low training efficiency, that is, long computation time taken to estimate numerous parameters. 
Method	Ridge 	Lasso	Polynomial 	GAM	Tree	Random Forest	SVM	Neural Network
Test MSE	6562.969	6563.246	5559.735	3409.622	14181.29	1279.8	1014.764	469.2368

 


Conclusion
In summary, our analysis of different prediction models for customer value reveals that flexible models like polynomials, GAM, and neural networks perform better than simpler models like Ridge and Lasso. Non-parametric models such as Random Forest and SVM outshine most parametric methods, showcasing their ability to capture complex relationships. Despite the drawbacks of individual regression trees, Random Forest proves effective by combining multiple trees.

The Neural Network emerges as the top-performing model, albeit with higher computational costs. 

References
Awad, M. K. (2015). Support vector regression. In Efficient learning machines: Theories, concepts, and applications for engineers and system designers (pp. 67-80).
Iranian Churn Dataset. (n.d.). Retrieved from UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
James, G. W. (2023). Support Vector Machines. In An introduction to statistical learning (pp. 367-403). Springer.


