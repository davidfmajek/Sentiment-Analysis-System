from dependencies import pd, np, re, stopwords, PorterStemmer, TfidfVectorizer, train_test_split, LogisticRegression, accuracy_score, nltk
import ssl, pickle
from stemming import stemming
from data_processing import model, X_test, Y_test

#User inputs
print("X sentiment Analysis\n")
tweet_index = int(input("Enter the index of the tweet you want to predict (0 to {}): ".format(len(X_test) - 1)))

# Ensuring the index is within the valid range
if 0 <= tweet_index < len(X_test):
    X_new = X_test[tweet_index]
    print("Actual label:", Y_test[tweet_index])  # Print the actual label

    # Make a prediction
    prediction = model.predict(X_new)

    # Display the result
    if prediction[0] == 0:
        print("Predicted label:", prediction[0])
        print("Negative Tweet")
    else:
        print("Predicted label:", prediction[0])
        print("Positive Tweet")
else:
    print("Invalid index. Please enter a number between 0 and {}.".format(len(X_test) - 1))