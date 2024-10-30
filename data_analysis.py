import pandas as pd

data = pd.read_csv('test.csv')

print(data.head())

# Check for missing values
df = data.copy()
df['tavg'].fillna(data['tavg'].mean(), inplace=True)
df['pres'].fillna(data['pres'].mean(), inplace=True)
df['wdir'].fillna(data['wdir'].mean(), inplace=True)

print(data.isnull().sum())
print(data.info())
print(data.describe())

