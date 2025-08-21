# import all packages and libraries used in this analysis
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# import the data set and convert to a data frame
data_scaled = pd.read_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D212\\Task 2\\df_scaled.csv')

# drop the index column
data_final = data_scaled.drop(data_scaled.columns[0], axis=1)
data_final.info()

# apply PCA
pca = PCA()
pca.fit(data_final)

# explained variance
explained_variance = pca.explained_variance_ratio_

print(explained_variance)

# initiate pca to get the matrix
principal_components = pca.fit_transform(data_final)

# convert to DataFrame for better readability
principal_components_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# compute the covariance matrix of the principal components
covariance_matrix = np.cov(principal_components_df.T)

# print the covariance matrix using Pandas for better formatting
print(pd.DataFrame(covariance_matrix, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])], index=[f'PC{i+1}' for i in range(principal_components.shape[1])]))

# display covariance matrix as a heatmap
sns.heatmap(covariance_matrix, annot=False, fmt=".2f", cmap="coolwarm", xticklabels=[f'PC{i+1}' for i in range(principal_components.shape[1])], yticklabels=[f'PC{i+1}' for i in range(principal_components.shape[1])])
plt.show()

# create a scree plot for PCA
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(1, len(explained_variance) + 1, step=1))
plt.grid(True)
plt.show()

# create cumulative explained variance chart
cumulative_explained_variance = np.cumsum(explained_variance)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(np.arange(1, len(cumulative_explained_variance) + 1, step=1))
plt.grid(True)
plt.show()

#define PCA model to use
pca = PCA(n_components=11)

#fit PCA model to data
pca_fit = pca.fit(data_final)

# create a DataFrame to display principal components and their explained variance ratios
principal_components_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(variance_each_pc))],
    'Explained Variance Ratio': variance_each_pc})

print(principal_components_df)

# sum the explained variance ratios of the first 11 principal components
variance_11_pcs = np.sum(explained_variance[:11])
print(variance_11_pcs)
