from dependencies import pd, np, re, stopwords, PorterStemmer, TfidfVectorizer, train_test_split, LogisticRegression, accuracy_score, nltk
import ssl

#Stemming is the process of reducing a word to its root or base form
#Example: running -> run| actor,actress,actor -> act.

port_stem = PorterStemmer()

def stemming(content):
    # Remove non-alphabetic characters and convert to lowercase
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    
    # Perform stemming and remove stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    
    # Join the stemmed words back into a single string
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content

