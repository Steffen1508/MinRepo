import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture

n_samples = 300
seed = 42

noisy_circles, labels_circles = datasets.make_circles(
    n_samples=n_samples,
    factor=0.5,
    noise=0.05,
    random_state=seed
)

blobs, labels_blobs = datasets.make_blobs(
    n_samples=n_samples,
    random_state=seed   
)

# Plot funktion
def plot_dataset(X, y, title):
    plt.figure(figsize=(6,6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plot alle tre datasæt
#plot_dataset(noisy_circles, labels_circles, "Noisy Circles")
#plot_dataset(blobs, labels_blobs, "Blobs")

#Opgave 1
blobs_data = pd.read_csv("blobs.csv")
X = blobs_data[blobs_data.columns[:-1]].values
labels_blobs = blobs_data[blobs_data.columns[-1]].values

# plot data
plt.figure(figsize=(6,6))
plt.scatter(X[:, 0], X[:, 1], c=labels_blobs, cmap='viridis')
plt.title("Blobs Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Scale features
scaler = StandardScaler()   
X_scaled = scaler.fit_transform(X)

k = 3 
kmeans = KMeans(n_clusters=k, random_state=seed)
kmeans_labels = kmeans.fit_predict(X_scaled)


dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Make a subplot of the ground truth, KMeans results and DBSCAN results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_blobs, cmap='viridis')
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].set_title("Original Data (true labels)")
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].set_title("KMeans Clustering")
axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
axes[2].set_xlabel("Feature 1")
axes[2].set_ylabel("Feature 2")
axes[2].set_title("DBSCAN Clustering")

plt.tight_layout()
plt.show()

# Load noisy circles dataset
circles_data = pd.read_csv('noisy_circles.csv')
X = circles_data[['Feature1', 'Feature2']].values
labels_circles = circles_data['labels'].values


# Scale features
scaler = StandardScaler()   
X_scaled = scaler.fit_transform(X)

# KMeans
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Make a subplot of the ground truth, KMeans results and DBSCAN results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_circles, cmap='viridis')
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].set_title("Original Data (true labels)")
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].set_title("KMeans Clustering")
axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
axes[2].set_xlabel("Feature 1")
axes[2].set_ylabel("Feature 2")
axes[2].set_title("DBSCAN Clustering")


plt.tight_layout()
plt.show()

