from dependencies import pd, np, re, stopwords, PorterStemmer, TfidfVectorizer, train_test_split, LogisticRegression, accuracy_score, nltk
import ssl, pickle
from stemming import stemming

# Naming the columns and reading the dataset 
column_names = ['target', 'id', 'date', 'flag', 'user', 'tweet']
twitter_data = pd.read_csv('/Users/davidmajek/Desktop/Python/X sentiment Analysis /twitter_data.csv', names=column_names, encoding='ISO-8859-1')

# Displaying dataset information
print("\nShape of the dataset:")
print(twitter_data.shape)

print("\nFirst 5 rows of the dataset:")
print(twitter_data.head(5))

# Counting missing values
print("\nNumber of missing values in the dataset:")
print(twitter_data.isnull().sum())

# Checking the distribution of the target column
print("\nDistribution of target values:")
print(twitter_data['target'].value_counts())

# Converting the target value "4" to "1" (0 -> negative, 1 -> positive)
twitter_data.replace({'target': {4: 1}}, inplace=True)