from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
data = scipy.io.loadmat("wine.mat")
X = data["X"]
y = data["y"].ravel()  # 0=normal, 1=anomali

bic_scores = []
n_range = range(1,20)

for n in n_range:
    gm = GaussianMixture(n_components=n, n_init=5, random_state=42).fit(X)
    bic_scores.append(gm.bic(X))

plt.plot(list(n_range), bic_scores, marker="o")
plt.xlabel("Antal Gaussianer")
plt.ylabel("BIC")
plt.title("BIC-kurve")
plt.show()

#Vuderingen er at n_components på 18 er bedst



x_filter = X[y==0]

gm = GaussianMixture(n_components=18, n_init=5, random_state=42).fit(x_filter)

"""
normal_data  = []
unormal_data = []
scores = gm.score_samples(X)
threshold = np.percentile(scores,12.9)
for i in scores:
    if i < threshold:
        unormal_data.append(i)
    else:
        normal_data.append(i)
slave måde
"""       
scores = gm.score_samples(X)
threshold = np.percentile(scores,12.9) 
y_pred = (scores < threshold).astype(int)
print(classification_report(y,y_pred))