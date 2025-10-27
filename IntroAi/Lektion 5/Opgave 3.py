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
housing = pd.read_csv('housing.csv')
housing.head()

num_features = housing.select_dtypes(include=[np.number]).columns
housing_num = housing[num_features]


scaler_std = StandardScaler()
housing_standard = pd.DataFrame(scaler_std.fit_transform(housing_num),columns=num_features)
print(housing_standard)

housing_standard.hist(bins=50, figsize=(20,15))
plt.show()

scaler_min_max = MinMaxScaler()
housing_min_max = pd.DataFrame(scaler_min_max.fit_transform(housing_num),columns=num_features)
print(housing_min_max)

housing_min_max.hist(bins=50, figsize=(20,15))
plt.show()

