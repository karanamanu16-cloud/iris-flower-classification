import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Iris Flower Classification")
print("---------------------------")

print("\nFeature Names:")
print(iris.feature_names)

print("\nFlower Classes:")
print(iris.target_names)

print("\nDataset Shape:")
print(X.shape)


# 2. Convert dataset into a DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

print("\nFirst 5 rows:")
print(df.head())


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# 4. Create the Machine Learning model
model = LogisticRegression(max_iter=200)

# 5. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# 8. Show classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 10. Test with a new flower
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = model.predict(new_flower)

predicted_flower = iris.target_names[prediction[0]]

print("\nNew Flower Prediction:")
print("Predicted flower:", predicted_flower)


# 11. Visualize the confusion matrix
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Iris Flower Classification - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    range(3),
    iris.target_names
)

plt.yticks(
    range(3),
    iris.target_names
)

plt.colorbar()

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.tight_layout()
plt.show()