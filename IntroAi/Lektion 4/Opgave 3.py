import sys
import matplotlib as mpl
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

housing = pd.read_csv('C:\\Users\\Raind\\Documents\\MinRepo\\IntroAi\\Lektion 4\\housing.csv')
housing.dropna(inplace=True)

housing.hist(bins=50, figsize=(20,15))

#Opgave kode

housing["rooms_cat"] = pd.cut(housing["total_rooms"],bins=[0., 5000, 10000, np.inf],labels=[0, 1, 2])
test_size = 0.01
 
# Random split
train_set, test_set = train_test_split(housing, test_size=test_size, random_state=42)
# Stratified split (brug samme kolonne)
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=test_size, stratify=housing["rooms_cat"], random_state=42)
 
print("Random split (test):")
print(test_set["rooms_cat"].value_counts(normalize=True))
 
print("Stratified split (test):")
print(strat_test_set["rooms_cat"].value_counts(normalize=True))
 
# 4) Fjern rooms_cat igen fra de datasæt hvis I vil fortsætte videre uden den
for ds in (train_set, test_set, strat_train_set, strat_test_set):
    ds.drop(columns=["rooms_cat"], inplace=True)


