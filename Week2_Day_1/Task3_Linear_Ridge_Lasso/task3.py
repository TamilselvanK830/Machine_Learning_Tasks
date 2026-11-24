import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

print("\n===== DIABETES DATASET =====")
print("Dataset Shape:", X.shape)
print("Number of Features:", len(feature_names))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n===== DATA SPLITTING =====")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_predictions)
)

print("\n===== LINEAR REGRESSION =====")
print("Test RMSE:", round(linear_rmse, 4))

linear_coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": linear_model.coef_
})

top_five = linear_coefficients.reindex(
    linear_coefficients["Coefficient"].abs()
    .sort_values(ascending=False).index
).head(5)

print("\n===== TOP 5 LINEAR REGRESSION COEFFICIENTS =====")
print(top_five.to_string(index=False))

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

ridge_predictions = ridge_model.predict(X_test)

ridge_rmse = np.sqrt(
    mean_squared_error(y_test, ridge_predictions)
)

ridge_zero_count = np.sum(ridge_model.coef_ == 0)

print("\n===== RIDGE REGRESSION =====")
print("Alpha:", 1.0)
print("Test RMSE:", round(ridge_rmse, 4))
print("Zero Coefficients:", ridge_zero_count)

lasso_model = Lasso(alpha=0.1, max_iter=100000)
lasso_model.fit(X_train, y_train)

lasso_predictions = lasso_model.predict(X_test)

lasso_rmse = np.sqrt(
    mean_squared_error(y_test, lasso_predictions)
)

lasso_zero_count = np.sum(lasso_model.coef_ == 0)

print("\n===== LASSO REGRESSION =====")
print("Alpha:", 0.1)
print("Test RMSE:", round(lasso_rmse, 4))
print("Zero Coefficients:", lasso_zero_count)

linear_zero_count = np.sum(linear_model.coef_ == 0)

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression"],
    "Test RMSE": [
        linear_rmse,
        ridge_rmse,
        lasso_rmse
    ],
    "Number of Zero Coefficients": [
        linear_zero_count,
        ridge_zero_count,
        lasso_zero_count
    ]
})

print("\n===== MODEL COMPARISON =====")
print(comparison.to_string(
    index=False,
    formatters={"Test RMSE": "{:.4f}".format}
))

coefficient_comparison = pd.DataFrame({
    "Feature": feature_names,
    "Linear": linear_model.coef_,
    "Ridge": ridge_model.coef_,
    "Lasso": lasso_model.coef_
})

print("\n===== COEFFICIENT COMPARISON =====")
print(coefficient_comparison.to_string(index=False))

print("\n===== LASSO ZERO COEFFICIENT EXPLANATION =====")
print("Lasso uses L1 regularisation.")
print("It can reduce less useful feature coefficients exactly to zero.")
print("This helps with feature selection and creates a simpler model.")

print("\n===== HOW TO INTERPRET A COEFFICIENT =====")
print("1. A positive coefficient means the prediction increases as that feature increases.")
print("2. A negative coefficient means the prediction decreases as that feature increases.")
print("3. A larger absolute coefficient means a stronger effect per feature unit,")
print("   when other features are held constant.")

