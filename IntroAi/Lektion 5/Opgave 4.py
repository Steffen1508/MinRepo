import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV


# Indlæs datasæt
housing = pd.read_csv('C:\\Users\\Raind\\Documents\\MinRepo\\IntroAi\\Lektion 5\\housing.csv')
housing.head()
housing.dropna(inplace=True)
num_features = housing.select_dtypes(include=[np.number]).columns

model = LinearRegression()

X = housing[num_features].drop("median_house_value", axis=1)
y = housing["median_house_value"]

scores = cross_val_score(model, X, y, cv=10, scoring="r2")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())