# Fake Review Detector

## Overview

This project is a simple machine learning application that tries to identify whether a review is real or fake. The idea came from observing how online platforms often contain exaggerated or misleading reviews, which makes it difficult for users to trust ratings.

The model is trained on a small dataset of labeled reviews and uses basic NLP techniques to classify input text.

---

## Problem Statement

Online reviews play an important role in decision-making, but many of them are not genuine. Fake reviews are often overly positive, repetitive, or written in an unnatural way. This project attempts to detect such reviews automatically.

---

## Approach

The project follows a straightforward pipeline:

1. Clean and preprocess the review text
2. Convert text into numerical form using TF-IDF
3. Train a classification model (Logistic Regression)
4. Predict whether a review is real or fake

---

## Features

* Takes review input from the terminal
* Classifies text as real or fake
* Handles noisy text (capital letters, repeated words, punctuation)
* Lightweight and easy to run

---

## Tech Stack

* Python
* pandas
* scikit-learn
* nltk

---

## Project Structure

fake-review-detector/
│
├── data/
├── model/
├── train.py
├── app.py
├── utils.py
├── README.md
└── requirements.txt

---

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train the model

python train.py

(This will create model.pkl and vectorizer.pkl inside the model/ folder)

### 3. Run the application

python app.py

---

## Example

Input:
THIS PRODUCT IS AMAZING!!! BUY NOW!!!

Output:
fake

---

## Challenges Faced

* Working with very small and noisy dataset
* Handling repeated characters and excessive punctuation
* Making sure preprocessing is consistent during training and prediction

---

## What I Learned

* Basics of text preprocessing in NLP
* How TF-IDF works for feature extraction
* Training and saving machine learning models
* Importance of clean and structured code

---

## Future Improvements

* Add a web interface for better usability
* Use a larger and more realistic dataset
* Try advanced models for better accuracy
* Add a neutral or uncertain category

---

This project was developed as part of a course assignment.
