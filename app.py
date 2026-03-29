# app.py

import pickle
from utils import preprocess_text

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_review(text):
    text = preprocess_text(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# CLI
while True:
    text = input("Enter review: ")
    result = predict_review(text)
    print("Prediction:", result)
