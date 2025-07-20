# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import joblib

df = pd.read_csv('data/iris.csv')
df = df.drop(columns=['Id'])
df['Species'] = df['Species'].str.replace('Iris-','',regex = False)

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# save the model
joblib.dump(model, 'model/iris_model.pkl')

print("Model trained and saved successfully.")