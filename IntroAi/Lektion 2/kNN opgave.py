import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
import sklearn as sk

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:\\Users\\Raind\\Documents\\MinRepo\\IntroAi\\iris.csv')

df.info()

df.describe()

df.dropna(inplace=True)

X=df[["sepal_length", "sepal_width"]]
y=df.target

#Split the dataset into traning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

class SimpleKNNClassifier:
    def __init__(self, n_neighbors=3):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        self.n_neighbors = n_neighbors
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = np.asarray(X_train)
        self._y_train = np.asarray(y_train).ravel()

        if self._X_train.shape[0] != self._y_train.shape[0]:
            raise ValueError("X_train and y_train must contain the same number of samples")

        return self

    def predict(self, X):
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("The classifier must be fitted before calling predict.")

        X = np.asarray(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        # Compute Euclidean distances to all training points
        distances = np.linalg.norm(self._X_train - x, axis=1)

        # Find the indices of the k nearest neighbors
        neighbor_indices = np.argsort(distances)[: self.n_neighbors]
        neighbor_labels = self._y_train[neighbor_indices]

        # Majority vote among nearest neighbor labels
        values, counts = np.unique(neighbor_labels, return_counts=True)
        majority_index = np.argmax(counts)
        return values[majority_index]


#Create a kNN classifier with a specified calue of k (e.g., k=3)
k = 3
knn_classifier = SimpleKNNClassifier(n_neighbors=k)

#Fit the classifier to the traning data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

#Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of kNN with k={k}: {accuracy * 100:.2f}%")

#resten er lavet af chatten
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Visualisering ---
h = 0.1  # grid step size
x_min, x_max = X["sepal_length"].min() - 1, X["sepal_length"].max() + 1
y_min, y_max = X["sepal_width"].min() - 1, X["sepal_width"].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

# Forudsig på hele grid'et
Z = knn_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Baggrund (decision boundary)
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)

# Træningspunkter
plt.scatter(
    X_train["sepal_length"], X_train["sepal_width"],
    c=y_train, cmap=cmap_bold, edgecolor="k", marker="o", s=50, label="Train"
)

# Testpunkter
plt.scatter(
    X_test["sepal_length"], X_test["sepal_width"],
    c=y_test, cmap=cmap_bold, edgecolor="k", marker="^", s=70, label="Test"
)

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title(f"kNN Decision Boundary (k={k}) • Accuracy={accuracy*100:.2f}%")
plt.legend()
plt.show()
