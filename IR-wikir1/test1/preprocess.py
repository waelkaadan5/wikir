import pandas as pd
import numpy as np
import string
import re
import nltk
import math
import pickle
import ir_datasets
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer, ISRIStemmer

dataset = ir_datasets.load("antique/train")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def remove_special_chars(sentence):
    """Remove special characters from the sentence"""
    special_chars = r'[.,!?@$%&()]'
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


def preprocess_docs():
    print("preeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    preprocessed_docs = {}
    for doc in dataset.docs_iter()[:2000]:
        preprocessed_text = preprocess(doc.text)
        preprocessed_docs[doc.doc_id] = preprocessed_text
    pickle.dump(preprocessed_docs, open("preprocessed_docs" + ".pickle", "wb"))


def preprocess_queries():
    preprocessed_queries = {}
    for query in dataset.queries_iter()[:20]:
        preprocessed_text = preprocess(query.text)
        preprocessed_queries[query.query_id] = preprocessed_text
    pickle.dump(preprocessed_queries, open("preprocessed_queries" + ".pickle", "wb"))


preprocess_docs()
preprocess_queries()
