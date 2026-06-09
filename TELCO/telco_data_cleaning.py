import pandas as pd
import numpy as np

df = pd.read_csv('TELCO\WA_Fn-UseC_-Telco-Customer-Churn.csv')
# print(df.info())
print(df.isna().sum())
print(df.corr(numeric_only=True))

# df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
# print(df[df['TotalCharges'] == ' '] )
# df[df['TotalCharges'] == ' ']
# print(df[df.eq('').any(axis=1)])
df = df.replace(' ', np.nan)
df = df.dropna()
# print(df.info())

df.to_csv("New_clean_telco", index=False)