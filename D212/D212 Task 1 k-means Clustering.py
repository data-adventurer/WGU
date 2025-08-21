# import all packages and libraries used in this analysis
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# import the data set and convert to a data frame
data_transformed = pd.read_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D212\\Data\\df_transformed.csv')

# determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_transformed)
    wcss.append(kmeans.inertia_)

# plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Vitamin D Levels')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# perform k-means clustering with the optimal number of clusters (2)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(data_transformed)

# add the cluster labels to the original dataframe
data_transformed['Cluster'] = clusters

# calculate the silhouette score
silhouette_avg = silhouette_score(data_transformed, clusters)
silhouette_avg

# visualize the clusters using seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_transformed, x='VitD_levels', y='VitD_levels', hue='Cluster', palette='viridis')
plt.title('K-Means Clustering of Vitamin D Levels')
plt.xlabel('Vitamin D Levels')
plt.ylabel('Vitamin D Levels')
plt.show()

# calculate the silhouette score
silhouette_avg = silhouette_score(data_transformed, clusters)
print(silhouette_avg)

# count the number of data points in each cluster
cluster_sizes = data_transformed['Cluster'].value_counts()
print(cluster_sizes)
