
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

"""
Load Dataset
"""
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Classes:", iris.target_names)
print("First 5 rows:\n", X[:5])

"""
Scalar features
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Before scaling:", X[0])
print("After scaling:", X_scaled[0])

"""
Split data 80/20
"""
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

"""
Train KNN model
"""
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print("Model trained successfully!")
print("Training on", X_train.shape[0], "samples")

"""
Make Predictions
"""
predictions = model.predict(X_test)

print("Prediction:", predictions)
print("Actual:    ", y_test)

""""
Evaluate the model
"""
print("\n=== Classification Report ===")
print(classification_report(y_test, predictions, target_names=iris.target_names))

f1 = f1_score(y_test, predictions, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")

"""
Confusion Matrix
"""
cm = confusion_matrix(y_test,predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix — Iris KNN Classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved!")