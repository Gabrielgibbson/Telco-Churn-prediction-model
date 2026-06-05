from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("Clean_Telco_dataset.csv")

X = df.drop(['Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Checking for which model will give the best accuracy
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions) * 100

print(predictions)
print(y_test)
print(accuracy)

model_filename = 'TelcoChurnModel.pkl'
joblib.dump(model, model_filename)
print(f"model saved permanently {model_filename}")

