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
from pandas.plotting import scatter_matrix

housing = pd.read_csv("C:\\Users\\Raind\\Documents\\MinRepo\\IntroAi\\Lektion 4\\housing.csv")
housing.dropna(inplace=True)

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

plt.show()