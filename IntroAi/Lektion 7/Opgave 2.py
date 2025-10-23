# --- Exercise 2: Logistic Regression (Cardiovascular death) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

# 1. Indlæs datasæt
df = pd.read_csv("CVDdata.csv")

# 2. Træn logistisk model på blodtryk (BP) alene
X = df[["BP"]]
y = df["CVD"]

model1 = LogisticRegression()
model1.fit(X, y)

# 3. Visualisér sigmoidkurven
# Sigmoid viser sandsynligheden for CVD som funktion af blodtryk
x_vals = np.linspace(df["BP"].min(), df["BP"].max(), 200).reshape(-1, 1)
y_proba = model1.predict_proba(x_vals)[:, 1]

plt.plot(x_vals, y_proba, color="red")
plt.xlabel("Blood Pressure")
plt.ylabel("Probability of CVD")
plt.title("Sigmoidkurve for logistisk regression (BP)")
plt.show()

# 4. Evaluer modellen med ROC-kurve og AUC
y_pred_proba = model1.predict_proba(X)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot(color="darkorange", linewidth=2)
plt.title("ROC-kurve for CVD (BP-baseret model)")
plt.show()

auc_value = auc(fpr, tpr)
print("AUC:", round(auc_value, 3))

# 5. Træn ny model på både blodtryk og rygning
X2 = df[["BP", "Smoking"]]
model2 = LogisticRegression()
model2.fit(X2, y)

y_pred_proba2 = model2.predict_proba(X2)[:, 1]
fpr2, tpr2, _ = roc_curve(y, y_pred_proba2)
auc_value2 = auc(fpr2, tpr2)

print("AUC med BP + Smoking:", round(auc_value2, 3))
print("Ændring i AUC:", round(auc_value2 - auc_value, 3))
