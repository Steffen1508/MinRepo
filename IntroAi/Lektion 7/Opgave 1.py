# --- Exercise 1: Linear Regression (CO2 emission in cars) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Indlæs datasættet
df = pd.read_csv("CO2data.csv")

# 2. Visualisér data (for at se sammenhængen mellem fx vægt og CO2)
plt.scatter(df["Weight"], df["CO2"], color="blue", label="Data")
plt.xlabel("Weight (kg)")
plt.ylabel("CO2 emission (g/km)")
plt.title("Sammenhæng mellem bilvægt og CO2-udledning")
plt.legend()
plt.show()

# 3. Træn en lineær regressionsmodel
X = df[["Weight", "Volume"]]  # to features
y = df["CO2"]

model = LinearRegression()
model.fit(X, y)

# Udskriv koefficienter (for at se hvilken feature betyder mest)
print("Koefficienter:", model.coef_)
print("Intercept:", model.intercept_)

# Sammenlign individuelle features:
# -> Den med størst koefficient påvirker CO2 mest

# 4. Forudsig CO2 for bil med vægt = 2300 kg og volume = 1300 cm³
pred = model.predict([[2300, 1300]])
print("Forudsagt CO2:", pred[0], "g/km")

# 5. Visualisér regressionslinjen for én feature (fx Weight)
#    (Kun for at illustrere modellen i 2D)
model_single = LinearRegression()
model_single.fit(df[["Weight"]], y)

plt.scatter(df["Weight"], y, color="lightblue", label="Data")
plt.plot(df["Weight"], model_single.predict(df[["Weight"]]),
         color="red", label="Regressionslinje")
plt.xlabel("Weight (kg)")
plt.ylabel("CO2 emission (g/km)")
plt.title("Lineær regression på vægt")
plt.legend()
plt.show()
