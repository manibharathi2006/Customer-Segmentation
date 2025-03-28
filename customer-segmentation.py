# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')

# Generate synthetic customer data
np.random.seed(42)
num_customers = 300

# Create dataframe with customer features
data = {
    'Age': np.random.randint(18, 70, size=num_customers),
    'Annual_Income': np.random.randint(20, 200, size=num_customers) * 1000,
    'Spending_Score': np.random.randint(1, 100, size=num_customers),
    'Purchase_Frequency': np.random.randint(1, 20, size=num_customers),
    'Online_Time': np.random.uniform(0.5, 5.0, size=num_customers)
}

df = pd.DataFrame(data)

# Feature selection for clustering
features = ['Annual_Income', 'Spending_Score', 'Online_Time']
X = df[features]

# Data preprocessing - Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow Method
wcss = []
silhouette_scores = []
cluster_range = range(2, 8)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cluster_range, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

# Based on the plots, select optimal number of clusters (let's choose 4)
optimal_clusters = 4

# Perform K-Means clustering
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_

# Analyze clusters
cluster_stats = df.groupby('Cluster')[features].mean()
cluster_stats['Count'] = df['Cluster'].value_counts().sort_index()

print("\nCluster Statistics:")
print(cluster_stats)

# Visualize clusters in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'blue', 'green', 'purple']
for i in range(optimal_clusters):
    cluster_data = df[df['Cluster'] == i]
    ax.scatter(cluster_data['Annual_Income'], 
               cluster_data['Spending_Score'], 
               cluster_data['Online_Time'], 
               c=colors[i], 
               label=f'Cluster {i}', 
               s=50, 
               alpha=0.6)

ax.set_xlabel('Annual Income')
ax.set_ylabel('Spending Score')
ax.set_zlabel('Online Time (hrs)')
ax.set_title('3D Visualization of Customer Segments')
plt.legend()
plt.show()

# 2D Visualization
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='Cluster', 
                palette=colors[:optimal_clusters], s=100, alpha=0.7)
plt.title('Customer Segmentation: Income vs Spending Score')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Profile each cluster
cluster_profiles = []

for i in range(optimal_clusters):
    cluster_data = df[df['Cluster'] == i]
    profile = {
        'Cluster': i,
        'Size': len(cluster_data),
        'Avg_Income': cluster_data['Annual_Income'].mean(),
        'Avg_Spending_Score': cluster_data['Spending_Score'].mean(),
        'Avg_Online_Time': cluster_data['Online_Time'].mean(),
        'Avg_Age': cluster_data['Age'].mean(),
        'Avg_Purchase_Freq': cluster_data['Purchase_Frequency'].mean()
    }
    cluster_profiles.append(profile)

profiles_df = pd.DataFrame(cluster_profiles)
profiles_df.set_index('Cluster', inplace=True)

print("\nCustomer Segment Profiles:")
print(profiles_df)

# Assign meaningful names to clusters based on characteristics
segment_names = {
    0: "High-Income Low Spenders",
    1: "Moderate-Income Moderate Spenders",
    2: "Low-Income High Spenders",
    3: "High-Income High Spenders"
}

df['Segment'] = df['Cluster'].map(segment_names)

# Final output with customer segments
print("\nSample of Customer Data with Segments:")
print(df[['Age', 'Annual_Income', 'Spending_Score', 'Online_Time', 'Segment']].head(10))
