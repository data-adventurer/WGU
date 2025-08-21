# import the packages and libraries needed for cleaning
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# create a link to the csv file in computer folders and create dataframe
# print the first five rows of the dataframe to ensure link works
medical_df = pd.read_csv('medical_raw_data.csv')
print(medical_df.head())

# change the data type of columns
medical_df['CaseOrder'] = medical_df['CaseOrder'].astype(str)
medical_df['Zip'] = medical_df['Zip'].astype(str)
medical_df['Lat'] = medical_df['Lng'].astype(str)
medical_df['Lng'] = medical_df['Lng'].astype(str)
medical_df['Area'] = medical_df['Area'].astype('category')
medical_df['Timezone'] = medical_df['Timezone'].astype('category')
medical_df['Employment'] = medical_df['Employment'].astype('category')
medical_df['Marital'] = medical_df['Marital'].astype('category')
medical_df['Gender'] = medical_df['Gender'].astype('category')
medical_df['Initial_admin'] = medical_df['Initial_admin'].astype('category')
medical_df['Services'] = medical_df['Services'].astype('category')

# change the 1's and 0's in Overweight and Anciety columns to Yes and No like other medical issue columns
medical_df['Overweight'] = medical_df['Overweight'].replace(0, 'No')
medical_df['Overweight'] = medical_df['Overweight'].replace(1, 'Yes')

medical_df['Anxiety'] = medical_df['Anxiety'].replace(0, 'No')
medical_df['Anxiety'] = medical_df['Anxiety'].replace(1, 'Yes')

# fill null values
medical_df['Children'] = medical_df['Children'].fillna(0)
medical_df['Soft_drink'] = medical_df['Soft_drink'].fillna('No')
medical_df['Overweight'] = medical_df['Overweight'].fillna('No')
medical_df['Anxiety'] = medical_df['Anxiety'].fillna('No')

# change the data types of children 
medical_df['Children'] = medical_df['Children'].astype(int)

# create a subset of the data frame to only include numerical columns
medical_numeric = medical_df[['Population', 'Children', 'Age', 'Income', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten', 'VitD_supp', 'Initial_days', 'TotalCharge', 'Additional_charges', 'Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8']]

# use k-nearest neighbour to impute values
medical_knn = medical_numeric.copy(deep=True)
knn_imputer = KNNImputer()
medical_knn.iloc[:, :] = knn_imputer.fit_transform(medical_knn)

medical_df['Age'] = medical_knn['Age']
medical_df['Income'] = medical_knn['Income']
medical_df['Initial_days'] = medical_knn['Initial_days']

# change the data types of age
medical_df['Age'] = medical_df['Age'].astype(int)
print(medical_df.info())

# review numerical columns to see how changes affected average and standard deviation
print(medical_numeric.describe())

# display the outliers through histrograms for investigation
population_df = medical_numeric['Population']
population_df_min = population_df[population_df<33894]
population_df_max = population_df[population_df<122814]

ax1 = sns.histplot(population_df_min)
ax1 = sns.histplot(population_df_max)
plt.show()

children_df = medical_numeric['Children']
children_df_min = children_df[children_df<8]
children_df_max = children_df[children_df<10]

ax2 = sns.histplot(children_df_min)
ax2 = sns.histplot(children_df_max,alpha=0.5)
plt.show()

income_df = medical_numeric['Income']
income_df_min = income_df[income_df<95749.6]
income_df_max = income_df[income_df<207249.13]

ax3 = sns.histplot(income_df_min)
ax3 = sns.histplot(income_df_max,alpha=0.5)
plt.show()

vitD_df = medical_numeric['VitD_levels']
vitD_df_min = vitD_df[vitD_df<9.519]
vitD_df_max = vitD_df[vitD_df<53.819]

ax4 = sns.histplot(vitD_df_min)
ax4 = sns.histplot(vitD_df_max,alpha=0.5)
plt.show()

# save the cleaned data frame to desktop
medical_df.to_csv('LorraineF_D206_CleanedData.csv')
