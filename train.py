# train.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import preprocess_text

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Preprocess
data["text"] = data["text"].apply(preprocess_text)

X = data["text"]
y = data["label"]

# TF-IDF
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression()
model.fit(X_vec, y)

# Save
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model trained!")
