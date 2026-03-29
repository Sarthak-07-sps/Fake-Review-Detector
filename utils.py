# utils.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def remove_repeated_chars(text):
    return re.sub(r'(.)\1+', r'\1\1', text)

def remove_stopwords(text):
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

def stem_text(text):
    return " ".join([stemmer.stem(w) for w in text.split()])

def preprocess_text(text):
    text = clean_text(text)
    text = remove_repeated_chars(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text
