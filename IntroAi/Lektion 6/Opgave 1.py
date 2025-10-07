import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, RocCurveDisplay
)

breast_cancer = pd.read_csv("C:\\Users\\Raind\\Documents\\MinRepo\\IntroAi\\Lektion 6\\breast-cancer.csv")

breast_cancer["diagnosis"] = breast_cancer["diagnosis"].map({"M": 1, "B": 0})

colors = breast_cancer["diagnosis"].map({0:"Blue",1:"Red"})

attributes = ["radius_mean","texture_mean","perimeter_mean","area_mean"]

'''
scatter_matrix(breast_cancer[attributes],figsize=(12,8),c=colors)
plt.show()
'''

X = breast_cancer.drop(columns=["id","diagnosis"])
X = X[attributes]
y = breast_cancer["diagnosis"]

print(f"Features: ",X.shape, "Outcome: ",y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=69,stratify=y)

scaler=StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.show()

