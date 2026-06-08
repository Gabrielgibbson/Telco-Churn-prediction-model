from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
# from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import joblib
# from scipy.stats import chi2_contingency

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv(r'STUDENT\ai_student_impact_dataset (1).csv')
# print(df.info())

texts = ['Burnout_Risk_Level','Major_Category','Year_of_Study', 'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Paid_Subscription', 'Institutional_Policy']

encoders = {}
for column in texts:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  encoders[column] = le


X = df.drop(['Student_ID', 'Post_Semester_GPA', 'Primary_Use_Case', 'Paid_Subscription', 'Burnout_Risk_Level'], axis=1)
y_GPA = df['Post_Semester_GPA']
y_burn = df['Burnout_Risk_Level']

# print(X)
X_train, X_test, y_train_GPA, y_test_GPA, y_train_burn, y_test_burn  = train_test_split(X, y_GPA, y_burn, test_size=0.2, random_state=42)
# model = RandomForestRegressor()
model1 = XGBRegressor()
model2 = XGBClassifier()
# model2 = RandomForestClassifier()
model1.fit(X_train, y_train_GPA)
model2.fit(X_train, y_train_burn)
prediction1 = model1.predict(X_test)
prediction2 = model2.predict(X_test)
error = mean_absolute_error(y_test_GPA, prediction1)
accuracy = accuracy_score(y_test_burn, prediction2)
# print(error)
# print(accuracy)

# joblib.dump(model1, 'GPA_predictor.pkl')
# joblib.dump(model2, 'Burnout_risk_level.pkl')
joblib.dump(model1, "GPA_predictor.pkl")
joblib.dump(model2, "Burnout_risk_level.pkl")

# save ALL encoders
joblib.dump(encoders, "encoders.pkl")
print('All models saved successfully')

# clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(X_train, X_test, y_train, y_test)
# print(models)


# corr = df.corr(numeric_only= True)
# print(corr["Post_Semester_GPA"].sort_values(ascending= False))