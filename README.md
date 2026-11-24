# Week 2 – Day 1: Machine Learning

## Overview
This folder contains four Machine Learning tasks completed using Python and Scikit-learn.

## Tasks

### Task 1 – Splits and Overfitting
- Loaded the Breast Cancer dataset.
- Split data into training, validation, and testing sets.
- Trained Decision Tree models with different depths.
- Compared training and validation accuracy.
- Visualized overfitting using a graph.

### Task 2 – Cross-Validation and Data Leakage
- Applied 5-Fold Cross-Validation.
- Trained Logistic Regression.
- Demonstrated data leakage using StandardScaler.
- Compared incorrect and correct scaling methods.
- Learned supervised and unsupervised learning.

### Task 3 – Linear, Ridge and Lasso Regression
- Loaded the Diabetes dataset.
- Split data into 80% training and 20% testing.
- Trained Linear Regression, Ridge, and Lasso models.
- Compared coefficients and test RMSE.
- Counted zero coefficients using Lasso.

### Task 4 – Logistic Regression and Probabilities
- Loaded the Breast Cancer dataset.
- Trained binary Logistic Regression.
- Used `predict_proba()` to calculate probabilities.
- Compared decision thresholds 0.5 and 0.3.
- Trained multi-class Logistic Regression using the Iris dataset.
- Displayed probability tables.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Installation

Install the required libraries:

```bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter ipykernel
```

## How to Run

Run Python files:

```bash
python task1.py
python task2.py
python task3.py
python task4.py
```

For notebook tasks, open the `.ipynb` files in Jupyter Notebook or VS Code and select the correct Python environment.

## Learning Outcomes
- Understanding training, validation, and testing data.
- Learning cross-validation and data leakage.
- Understanding Linear, Ridge, and Lasso Regression.
- Learning Logistic Regression and prediction probabilities.
- Evaluating Machine Learning models.

## Author
Tamil

## Status
All four tasks completed.
