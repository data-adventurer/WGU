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
data = pd.read_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D212\\Data\\medical_clean.csv')

# check for null values
print(data.isnull().values.any())

# check for duplicates
duplicates = len(data) - len(data.drop_duplicates())

print('There are ' + str(duplicates) + ' duplicates in this data set')

# keep only numerical fields
data_clean = data.select_dtypes(include=['number'])
data_clean.head()

# keep only quantitative fields
data_final = data_clean.drop(['CaseOrder', 'Zip', 'Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8'], axis=1)
data_final.info()

# standardize the data
scaler = StandardScaler()
data_scaled = data_final.copy()

# create a dataframe with the scaled data
data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled), columns=data_scaled.columns)

data_scaled.head()

# export to csv
data_scaled.to_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D212\\Task 2\\df_scaled.csv')
