#MM Lektion 2

#Test om lortet virker

import sys

import pandas as pd
import sklearn as sk
import matplotlib as mpl

print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-learn {sk.__version__}")
print(f"Matplotlib{mpl.__version__}")

# Basic intro til Pandas
#1. Read Data
df = pd.read_csv('data.csv')

#2. Head & Tails
df.head() #first 5 rows
df.head() #last 5 rows

#3.Info
df.info()

#4. Descriptive statistics
df.describe()

#5. Handling missing values
df_dropna = df.propna()

df_fillna = df.fillna(0) #filling missing

#6. Selecting Colums
selected_colums = df[['column1','column2']]

#7. filtering

filtered_df = df[df['column1']>10]

#8 Grouping

grouped_df = df.groupby('column').agg({'column2':'mean'})

#Cheat sheet https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

