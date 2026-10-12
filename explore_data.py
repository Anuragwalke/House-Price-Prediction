import pandas as pd

# load the housing dataset
df = pd.read_csv("Housing.csv")

#display the first five rows
print("first 5 rows:")
print(df.head())

#display teh dataset size
print("\nDataset shape:")
print(df.shape)

# display column names and data types
print("\nDataset information:")
df.info()

#check for missing values
print(df.isnull().sum())