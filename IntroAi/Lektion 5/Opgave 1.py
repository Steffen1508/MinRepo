import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score


# Indlæs datasæt
housing = pd.read_csv('C:\\Users\\Raind\\Documents\\MinRepo\\IntroAi\\Lektion 5\\housing.csv')
housing.head()

print(len(housing))

print(housing.isnull().sum())

num_features = housing.select_dtypes(include=[np.number]).columns

print(num_features)

imputer_zero = SimpleImputer(strategy="constant",fill_value=0)
imputer_mean = SimpleImputer(strategy="mean")
imputer_median = SimpleImputer(strategy="median")

housing_zero = pd.DataFrame(imputer_zero.fit_transform(housing[num_features]),columns=num_features)
housing_mean = pd.DataFrame(imputer_mean.fit_transform(housing[num_features]),columns=num_features)
housing_median = pd.DataFrame(imputer_median.fit_transform(housing[num_features]),columns=num_features)

knn_imputer = KNNImputer(n_neighbors=5)
housing_knn = pd.DataFrame(knn_imputer.fit_transform(housing[num_features]),columns=num_features)


#ChatGPT kode
# col = "total_bedrooms"  # vælg den kolonne hvor du har NaN

# fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# # Zero
# axes[0].hist(housing_zero[col], bins=50, color="skyblue")
# axes[0].set_title("Zero Imputer")

# # Mean
# axes[1].hist(housing_mean[col], bins=50, color="lightgreen")
# axes[1].set_title("Mean Imputer")

# # Median
# axes[2].hist(housing_median[col], bins=50, color="salmon")
# axes[2].set_title("Median Imputer")

#plt.tight_layout()

#Plotting af histiorgram
housing_median.hist(bins=50, figsize=(20,15))
plt.show()
