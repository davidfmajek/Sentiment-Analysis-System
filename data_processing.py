# %%
from dependencies import pd, np, re, stopwords, PorterStemmer, TfidfVectorizer, train_test_split, LogisticRegression, accuracy_score, nltk
import ssl, pickle
from stemming import stemming

# Naming the columns and reading the dataset 
column_names = ['target', 'id', 'date', 'flag', 'user', 'tweet']
twitter_data = pd.read_csv('/Users/davidmajek/Desktop/Python/X sentiment Analysis /twitter_data.csv', names=column_names, encoding='ISO-8859-1')

# Converting the target value "4" to "1" (0 -> negative, 1 -> positive)
twitter_data.replace({'target': {4: 1}}, inplace=True)

# Stemming the entire tweet column in the dataset
twitter_data['stemmed_content'] = twitter_data['tweet'].apply(stemming)

# Displaying the first 5 rows of the dataset with the 'stemmed_content' column
print(twitter_data.head())
print(twitter_data['stemmed_content'])

# Separating the data and the labels
X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Converting textual data to numerical data
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("\nShape of X_train:", X_train.shape)
print("\nShape of X_test:", X_test.shape)

# Training the Machine Learning Model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Model Evaluation (Accuracy score on the training data)
print(">>>>>> TRAINING DATA <<<<<<")
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on the training data: ', training_data_accuracy)

# %%
# Model Evaluation (Accuracy score on the test data)
print(">>>>>> TEST DATA <<<<<<")
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on the test data: ', test_data_accuracy)

# Based on the score of the training and test data we can conclude that the model has performed well
# Model accuracy = 77.7%


# %%
#final step
X_new = X_test[200] #[200] is the 200th tweet in the dataset
print(Y_test[200])  

prediction = model.predict(X_new)
if(prediction[0] == 0):
    print(prediction)
    print("Negative Tweet")
else:
    print(prediction)
    print("Positive Tweet")
