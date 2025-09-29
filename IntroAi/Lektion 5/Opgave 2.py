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

print(housing["ocean_proximity"].isnull().sum())

#Der er ikke missing values
num_features = housing.select_dtypes(include=[np.number]).columns


housing["ocean_proximity"].value_counts()
housing["ocean_proximity"] = housing["ocean_proximity"].replace("Na", np.nan)

#Label encoding

ordinal_encoder = OrdinalEncoder()
housing["ocean_proximity_encoded"] = ordinal_encoder.fit_transform(housing[["ocean_proximity"]])


print(housing["ocean_proximity_encoded"])

#One-hot encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
ocean_1hot = onehot_encoder.fit_transform(housing[["ocean_proximity"]]) #Her gemmer vi featuren i en variable
ocean_1hot_df = pd.DataFrame(ocean_1hot,columns=onehot_encoder.get_feature_names_out(["ocean_proximity"])) #her gør 1h sine tingz
print(ocean_1hot_df)


