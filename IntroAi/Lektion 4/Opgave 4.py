import sys
import matplotlib as mpl
import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import colormaps   


housing = pd.read_csv("C:\\Users\\Raind\\Documents\\MinRepo\\IntroAi\\Lektion 4\\housing.csv")
housing.dropna(inplace=True)

corr_matrix = housing.select_dtypes(include=[np.number]).corr()

print(corr_matrix["median_house_value"].sort_values(ascending=False))

plt.matshow(corr_matrix, cmap="coolwarm")
plt.colorbar()
plt.show()

# target_corr = corr_matrix[["median_house_value"]]

# plt.matshow(target_corr, cmap="coolwarm")
# plt.colorbar()
# plt.yticks(range(len(target_corr.index)), target_corr.index)
# plt.xticks([0], ["median_house_value"])
# plt.show()