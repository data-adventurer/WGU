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
data = pd.read_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D212\\Data\\medical_clean.csv')

# inspect the data frame
print(data.head())
print(data.info())

# check for null values
print(data.isnull().values.any())

# check for duplicates
duplicates = len(data) - len(data.drop_duplicates())

print('There are ' + str(duplicates) + ' duplicates in this data set')

# only include continuous variables
data_clean = data.select_dtypes(include=['float64'])

# normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clean)

print(data_scaled)

# convert numpy array to dataframe to export as csv
data_transformed = pd.DataFrame(data_scaled, columns=data_clean.columns)

# export to csv
data_transformed.to_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D212\\Data\\df_transformed.csv')
