from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer, ISRIStemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
import re

def remove_special_chars(sentence):
    """Remove special characters from the sentence"""
    special_chars = r'[.,!?@$%&()\'\"]'
    new_str = re.sub(special_chars, ' ', sentence)
    return new_str.split()


def remove_stop_Words(sentence):
    filtered_sentence = []
    filtered_sentence = [word for word in sentence if word not in stop_words]
    return filtered_sentence


def stemming(sentence):
    return [stemmer.stem(i) for i in sentence]


def lemmatization(sentence):
    return [lemmatizer.lemmatize(i) for i in sentence]


def preprocess(sentence, rm_stop_words=True, rm_special_chars=True, stemmer=True, lemmatizer=True):
    sentence = str(sentence).split()
    if rm_special_chars:
        sentence = remove_special_chars(" ".join(sentence))

    if stemmer:
        sentence = stemming(sentence)

    if lemmatizer:
        sentence = lemmatization(sentence)

    if rm_stop_words:
        sentence = remove_stop_Words(sentence)

    return " ".join(sentence)
