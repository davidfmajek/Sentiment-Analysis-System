This project is a Python-based sentiment analysis system designed to classify tweets as either positive or negative using Logistic Regression. It utilizes Natural Language Processing (NLP) techniques, including TF-IDF vectorization and stemming, to analyze sentiment. The model achieves 77.7% accuracy.
Features

Logistic Regression for sentiment classification
TF-IDF vectorization for feature extraction
NLTK for text preprocessing (tokenization, stemming)
Model persistence using pickle to save the trained model
Interactive sentiment prediction for user-provided tweets


Model Training

The model was trained using a dataset of tweets, with text preprocessing steps including tokenization, stemming, and TF-IDF vectorization. The Logistic Regression algorithm was employed for classification.
Dependencies

Pandas
NumPy
NLTK
scikit-learn
pickle
