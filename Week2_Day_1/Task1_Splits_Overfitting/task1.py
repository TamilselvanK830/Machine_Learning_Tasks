import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data
y = data.target

print("\n===== BREAST CANCER DATASET =====")
print("Dataset Shape:", X.shape)

print("\n===== CLASS BALANCE =====")
class_counts = pd.Series(y).value_counts().sort_index()

for class_number, count in class_counts.items():
    print(f"{data.target_names[class_number]} ({class_number}): {count}")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\n===== DATASET SPLIT =====")
print("Training Data:", X_train.shape[0])
print("Validation Data:", X_val.shape[0])
print("Testing Data:", X_test.shape[0])

print("\n===== SPLIT PERCENTAGE =====")

print("Training: 60%")
print("Validation: 20%")
print("Testing: 20%")


depths = [1, 3, 5, 10, None]

results = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Training prediction
    train_prediction = model.predict(X_train)

    # Validation prediction
    val_prediction = model.predict(X_val)

    # Calculate accuracy
    train_accuracy = accuracy_score(
        y_train, train_prediction
    )

    val_accuracy = accuracy_score(
        y_val, val_prediction
    )

    results.append({
        "Depth": str(depth),
        "Train accuracy": train_accuracy,
        "Validation accuracy": val_accuracy
    })

result_df = pd.DataFrame(results)

print("\n===== DEPTH VS ACCURACY =====")

print(
    result_df.to_string(
        index=False,
        formatters={
            "Train accuracy": "{:.4f}".format,
            "Validation accuracy": "{:.4f}".format
        }
    )
)

best_val_index = result_df["Validation accuracy"].idxmax()

best_val_accuracy = result_df.loc[
    best_val_index, "Validation accuracy"
]

overfit_index = None

for i in range(best_val_index + 1, len(result_df)):

    if (
        result_df.loc[i, "Train accuracy"]
        > result_df.loc[best_val_index, "Train accuracy"]
        and
        result_df.loc[i, "Validation accuracy"]
        <= best_val_accuracy
    ):
        overfit_index = i
        break

print("\n===== OVERFITTING ANALYSIS =====")

if overfit_index is not None:
    print(
        "Possible overfitting begins around depth:",
        result_df.loc[overfit_index, "Depth"]
    )
else:
    print("No clear overfitting point detected.")

plt.figure(figsize=(10, 6))

x_values = range(len(depths))

plt.plot(
    x_values,
    result_df["Train accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    x_values,
    result_df["Validation accuracy"],
    marker="s",
    label="Validation Accuracy"
)

plt.xticks(x_values, result_df["Depth"])

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")

plt.title("Decision Tree - Depth vs Accuracy")

plt.ylim(0.85, 1.02)
plt.grid(True)
plt.legend()

if overfit_index is not None:

    plt.axvline(
        x=overfit_index,
        color="red",
        linestyle="--",
        label="Possible Overfitting"
    )

    plt.annotate(
        "Possible Overfitting",
        xy=(
            overfit_index,
            result_df.loc[overfit_index, "Validation accuracy"]
        ),
        xytext=(overfit_index - 1.5, 0.88),
        arrowprops={"arrowstyle": "->", "color": "red"},
        color="red"
    )

plt.tight_layout()

# Save graph
plt.savefig("overfitting_plot.png")

plt.show()
print("Graph saved as: overfitting_plot.png")
