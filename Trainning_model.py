import pandas as pd
import re
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('SMSSpamCollection.txt', sep='\t', header=None, names=['Target', 'Text'])

def expand_abbreviations(senti):
    abbreviation_dict = {
        "dun": "do not",
        "don": "do not",
        "cant": "can not",
        "pl": "please",
        "dont": "do not"

    }
    for abbr, expanded in abbreviation_dict.items():
        senti = senti.replace(abbr, expanded)
    return senti


stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess and expand abbreviations for each sentence
sentis = []
for sen in data['Text']:
    senti = re.sub('[^A-Za-z]', ' ', sen)
    senti = senti.lower()
    words = word_tokenize(senti)
    word = [stem.stem(i) for i in words if i not in stop_words]
    senti = ' '.join(word)
    senti = expand_abbreviations(senti)
    sentis.append(senti)

data['Porter_Text_Process'] = sentis

cv=CountVectorizer(max_features=5000)
features = cv.fit_transform(data['Porter_Text_Process'])
features = features.toarray()

X = features
y = data['Target']

X_train = X
y_train = y

svm_model = SVC(C=0.1, class_weight='balanced', gamma='scale', kernel='linear', shrinking=True)
svm_model.fit(X_train, y_train)

with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)

with open('count_vectorizer.pkl', 'wb') as file:
    pickle.dump(cv, file)