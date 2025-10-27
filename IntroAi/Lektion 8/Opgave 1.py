import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

seed = random.randint(0,100)

df = pd.read_csv("titanic.csv")
#Data trimming, to only use the important data
df.head()
df.info()
df.drop(["PassengerId","Name","Ticket","Embarked","Cabin"],axis=1,inplace=True)
df.info()

df.replace(["male","female"],[0,1],inplace=True)
df.dropna(inplace=True)


X = df[["Pclass","Sex","Age"]]
y = df[["Survived"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=seed)


#Best min_sample_leaf check
#for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:

model = DecisionTreeClassifier(min_samples_leaf=9,max_features=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"max_feature = 2")
print(f"min_sample_leaf = 9")
print(f"Accuracy DecisionTree: {accuracy * 100:.2f}%")
#Best case for decisiontree was 81% accuracy

#Random Forrest

#for n in [1,2,3,4,5]:
model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy RandomForrest: {accuracy * 100:.2f}%")

