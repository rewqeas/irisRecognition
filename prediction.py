import joblib
# load the saved model
model = joblib.load('model/iris_model.pkl')

def predict(features):
    return model.predict([features])