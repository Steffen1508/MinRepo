import sys

import matplotlib as mpl
import pandas as pd
import sklearn as sk

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:\\Users\\Raind\\Documents\\MinRepo\\IntroAi\\Lektion 2\\iris.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.info()

df.describe()

df.dropna(inplace=True)

#X=df[["sepal_length", "sepal_width"]]
y=df.target
X = df.iloc[:, 0:3].values

#Split the dataset into traning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)
X_train_small, y_train_small = X_train[:50], y_train[:50]


#Create a kNN classifier with a specified calue of k (e.g., k=3)
k = 25
knn_classifier = KNeighborsClassifier(n_neighbors=k)

#Fit the classifier to the traning data
knn_classifier.fit(X_train_small, y_train_small)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)



#Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Train accuracy: {knn_classifier.score(X_train, y_train) * 100:.2f}%")
print(f"Accuracy of kNN with k={k}: {accuracy * 100:.2f}%")
print(k)




#resten er lavet af chatten

""" import numpy as np
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
 """