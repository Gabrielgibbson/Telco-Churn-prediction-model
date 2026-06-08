import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib


df = pd.read_csv("WineQT.csv")

print(df.isnull().sum())
print(df.drop_duplicates())
corr = df.corr()
print(corr)

X = df[['citric acid','density']]
y = df["fixed acidity"]
X_train,X_test, y_train,y_test = train_test_split(X, y, random_state = 50, test_size = 0.2)

# THE RANGE FOR DENSITY IS 0.99000 TO 1.0037
# THE RANGE FOR CITRIC ACID IS 0.0 TO 1.0


model = LinearRegression()
model.fit(X_train.values, y_train)
prediction = model.predict(X_test.values)
print(X_test)
print(prediction)

error = mean_squared_error(y_test, prediction)
print (error)

model_filename = "predicting_wine_acidity.pkl"
joblib.dump(model, model_filename)
print(f"model saved permanently {model_filename}")

