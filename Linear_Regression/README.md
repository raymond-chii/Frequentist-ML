# Regression Analysis on Cancer and Heart Disease Datasets

## Overview

This project involves applying various regression techniques (Linear, Ridge, and Lasso) to two datasets: a prostate cancer dataset and a South African heart disease dataset. The goal was to compare the performance of these methods, understand the relationships between various predictors and the target variables, and explore the impact of feature engineering.

## Datasets

Both datasets were obtained from [https://hastie.su.domains/ElemStatLearn/data.html](https://hastie.su.domains/ElemStatLearn/data.html).

1. Prostate Cancer Dataset
   - Target variable: lpsa (log of prostate-specific antigen)
   - Features: lcavol, lweight, age, lbph, svi, lcp, gleason, pgg45

2. South African Heart Disease Dataset
   - Target variable: chd (coronary heart disease)
   - Features: sbp, tobacco, ldl, adiposity, typea, obesity, alcohol, age

## Methods Applied

1. Linear Regression
2. Ridge Regression
3. Lasso Regression

For each method, the data was split into training, validation, and test sets. The models were trained on the training set, hyperparameters (for Ridge and Lasso) were tuned using the validation set, and final performance was evaluated on the test set.

## Results

### Prostate Cancer Dataset

1. Linear Regression
   - Test MSE: 0.4137
   - Most significant predictors: lcavol, svi, lweight

2. Ridge Regression
   - Best lambda: 2.3101
   - Test MSE: 0.4038
   - Test R2: 0.6788

3. Lasso Regression
   - Best alpha (lambda): 0.0008
   - Test MSE: 0.4156

### South African Heart Disease Dataset

1. Linear Regression
   - Test MSE: 0.1810
   - Most significant predictors: age, ldl, typea

2. Ridge Regression
   - Best lambda: 0.0001
   - Test MSE: 0.1810

3. Lasso Regression
   - Best alpha (lambda): 0.0095
   - Test MSE: 0.1797

### Feature Engineering (Prostate Cancer Dataset)

Custom features added:
- lcavol_squared: square of log cancer volume
- lcavol_pgg45: interaction between log cancer volume and percent of Gleason scores 4 or 5
- age_lbph: interaction between age and log of benign prostatic hyperplasia amount

Results with Ridge Regression:
- Original features: Test MSE: 0.4038, R2: 0.6788
- Custom features: Test MSE: 0.3664, R2: 0.7086

## Interpretation of Results

### Linear Regression

- For both datasets, linear regression provided interpretable results and identified the most significant predictors.
- In the cancer dataset, lcavol (log cancer volume), svi (seminal vesicle invasion), and lweight (log prostate weight) were the most important predictors.
- For the heart disease dataset, age, LDL cholesterol, and Type A behavior were the most significant predictors.

### Ridge Regression

- Ridge regression helped prevent overfitting, especially with the cancer dataset.
- The impact of regularization was more pronounced in the cancer dataset than in the heart disease dataset.
- With custom features, Ridge regression showed improved performance on the cancer dataset, indicating that the added features captured important nonlinear relationships.

### Lasso Regression

- Lasso regression performed well and offered feature selection capabilities.
- For the heart disease dataset, Lasso slightly outperformed the other methods, suggesting it effectively selected relevant features.

### Feature Engineering

- The addition of custom features (lcavol_squared, lcavol_pgg45, and age_lbph) to the cancer dataset improved the model's performance significantly.
- The improvement in both MSE and R2 indicates that these engineered features captured important nonlinear relationships and interactions between predictors.

## Interpretation of Graphs

1. Lasso Coefficient Profiles:
   - The graphs show how the coefficients of different features change as the regularization strength increases (moving from right to left).
   - Features whose lines stay at zero for longer are considered less important by the Lasso model.
   - For the cancer dataset, lcavol and lweight appear to be the most important features, as their coefficients remain non-zero even with strong regularization.
   - For the heart disease dataset, age seems to be the most important feature, followed by tobacco and ldl.

2. Ridge Regression Coefficients:
   - These graphs show how the standardized coefficients change with different levels of regularization (df(λ)).
   - As regularization increases (moving from right to left), coefficients are shrunk towards zero, but unlike Lasso, they typically don't become exactly zero.
   - The relative importance of features can be inferred from the magnitude of their coefficients at different regularization levels.

## Lessons Learned

1. Importance of Feature Selection: Lasso regression demonstrated its ability to automatically select relevant features, which can be particularly useful in high-dimensional datasets.

2. Value of Regularization: Both Ridge and Lasso regression showed how regularization can help prevent overfitting and improve model performance, especially in the cancer dataset.

3. Impact of Feature Engineering: The custom features added to the cancer dataset significantly improved model performance, highlighting the importance of domain knowledge and creative feature creation in predictive modeling.

4. Dataset Characteristics Matter: The effectiveness of different regression techniques varied between the two datasets, emphasizing the importance of trying multiple approaches and not relying on a one-size-fits-all solution.

5. Interpretability vs. Performance: While more complex models (like those with engineered features) can offer better performance, they may come at the cost of reduced interpretability. It's important to balance these factors based on the specific needs of the project.

6. Visualization Aids Understanding: The coefficient profile plots for Lasso and Ridge regression provided valuable insights into feature importance and the impact of regularization, demonstrating the value of visualization in understanding model behavior.

In conclusion, this project demonstrated the nuances of applying different regression techniques to real-world datasets. It highlighted the importance of feature engineering, the power of regularization, and the need for a thoughtful approach to model selection and evaluation.