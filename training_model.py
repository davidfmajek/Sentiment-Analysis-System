
from dependencies import pd, np, re, stopwords, PorterStemmer, TfidfVectorizer, train_test_split, LogisticRegression, accuracy_score, nltk
import ssl
from data_processing import model
import pickle


filename = "trained_model.sav"
pickle.dump(model, open(filename,'wb'))
print("training model saved")


#using the saved model FOR FUTURE PREDICTIONS.
#loaded_model = pickle.load(open('/Users/davidmajek/Desktop/Python/X sentiment Analysis /trained_model.sav','rb'))

