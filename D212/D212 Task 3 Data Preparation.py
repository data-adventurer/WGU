import pandas as pd

# load the dataset
df = pd.read_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D212\\Data\\medical_clean.csv')

# check for null values
print(df.isnull().values.any())

# check for duplicates
duplicates = len(df) - len(df.drop_duplicates())

print('There are ' + str(duplicates) + ' duplicates in this data set')

# list of medical conditions to consider
conditions = ['HighBlood', 'Stroke', 'Complication_risk', 'Overweight', 'Arthritis', 'Diabetes', 'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma']

# create a binary matrix for the conditions
basket = df[conditions].map(lambda x: 1 if x == 'Yes' else 0)

# display the first few rows of the binary matrix
basket.head()

# convert to boolean
basket_bool = basket.astype(bool)

# display dataframe
basket_bool.info()

# convert to csv file
basket_bool.to_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D212\\Task 3\\basket_data.csv', index=False)
