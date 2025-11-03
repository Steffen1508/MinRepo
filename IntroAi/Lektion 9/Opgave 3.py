import seaborn as sns
from pandas.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture

# Select features
df = pd.read_csv("Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inertia
k_values = range(2, 11) # Starter fra 2 (inklusiv) til 11 (eksklusiv) - Så vi får en list med k fra 2 til 10

inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(k_values, inertias, marker='o')
plt.title("Elbow Method (Inertia vs. Number of Clusters)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.grid(True)
plt.show()


# Silhouette Scores
sil_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)


# Plot Silhouette Score
plt.subplot(1,2,2)
plt.plot(k_values, sil_scores, marker='o', color='orange')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Choose best k
best_k = k_values[sil_scores.index(max(sil_scores))]
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Plot K-Means clusters
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans_labels, cmap='viridis')
plt.title(f"K-Means Clusters (k={best_k})")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()

print(kmeans_labels.shape)
print(df.shape)
df['cluster'] = kmeans_labels

groups = df.groupby('cluster')
for name, group in groups:
    print(f"Cluster {name}:")
    print(group[['Annual Income (k$)', 'Spending Score (1-100)']].describe())
    print()

# Boxplot for Annual Income
plt.figure(figsize=(8,6))
sns.boxplot(x='cluster', y='Annual Income (k$)', data=df)
plt.title("Annual Income per Cluster")
plt.show()

# Boxplot for Spending Score
plt.figure(figsize=(8,6))
sns.boxplot(x='cluster', y='Spending Score (1-100)', data=df)
plt.title("Spending Score per Cluster")
plt.show()

scatter_matrix(df[df.columns[:-1]], figsize=(10,10), diagonal='kde', c=df['cluster'], marker='o')
plt.show()
