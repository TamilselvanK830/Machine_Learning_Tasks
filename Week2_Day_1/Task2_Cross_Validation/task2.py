import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data = load_breast_cancer()

X = data.data
y = data.target

print("\n===== BREAST CANCER DATASET =====")
print("Dataset Shape:", X.shape)

model = LogisticRegression(max_iter=10000)

print("\n===== INCORRECT SCALING (DATA LEAKAGE) =====")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

leakage_scores = cross_val_score(
    model,
    X_scaled,
    y,
    cv=5
)

print("CV Scores:", np.round(leakage_scores, 4))
print("Mean:", round(leakage_scores.mean(), 4))
print("Standard Deviation:", round(leakage_scores.std(), 4))

print("\n===== CORRECT SCALING (NO DATA LEAKAGE) =====")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=10000))
])

correct_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5
)

print("CV Scores:", np.round(correct_scores, 4))
print("Mean:", round(correct_scores.mean(), 4))
print("Standard Deviation:", round(correct_scores.std(), 4))

print("\n===== DATA LEAKAGE COMPARISON =====")

print("Incorrect Mean:", round(leakage_scores.mean(), 4))
print("Correct Mean:", round(correct_scores.mean(), 4))

print(
    "Mean Score Difference:",
    round(leakage_scores.mean() - correct_scores.mean(), 4)
)

print("\n===== SUPERVISED VS UNSUPERVISED =====")

examples = {
    "Spam Detection": "Supervised Learning",
    "Customer Segmentation": "Unsupervised Learning",
    "House Price Prediction": "Supervised Learning",
    "Anomaly Detection": "Unsupervised Learning"
}

for task, learning_type in examples.items():
    print(f"{task}: {learning_type}")

print("\n===== MACHINE LEARNING WORKFLOW =====")

print("1. Collect: Gather data from different sources.")
print("2. Clean: Remove errors and handle missing values.")
print("3. Split: Divide data into training and testing sets.")
print("4. Train: Teach the model using training data.")
print("5. Evaluate: Check model performance.")
print("6. Improve: Tune the model to get better results.")
print("7. Deploy: Make the trained model available for use.")
