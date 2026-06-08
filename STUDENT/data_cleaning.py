import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv(r'STUDENT\ai_student_impact_dataset (1).csv')

# print(df.describe())
# print(df.info())

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
print(df.info())



# target = "Burnout_Risk_Level"
# texts = [col for col in df.select_dtypes(include=["object", 'category']).columns if col != target]
# print(texts)

# corr = df.corr(numeric_only= True)
# print(corr["Burnout_Risk_Level"].sort_values(ascending= False))
# print(data.isnull().sum())
# print(data.corr(numeric_only=True))