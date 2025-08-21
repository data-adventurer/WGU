import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

basket = pd.read_csv('C:\\Users\\LorraineFerrusi\\OneDrive - The Information Lab\\Documents\\Personal\\WGU\\D212\\Task 3\\basket_data.csv')

# convert to boolean
basket_bool = basket.astype(bool)

# display an example transaction (patient's medical conditions)
example_transaction = basket_bool.iloc[0]
print(example_transaction)

# generate frequent itemsets
frequent_itemsets = apriori(basket_bool, min_support=0.1, use_colnames=True)

# gisplay the frequent itemsets
print(frequent_itemsets)

# generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# display the rules
print(rules)

# display the summary statistics for the support, confidence, and lift
print(rules[['support', 'confidence', 'lift']].describe())

# display the top three relevant rules based on lift
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].nlargest(3, 'lift'))

