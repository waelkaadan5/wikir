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

dataset = ir_datasets.load("wikir/en1k/training")
from preprocess_function import preprocess

def preprocess_docs():
    preprocessed_docs = {}
    for doc in dataset.docs_iter():
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
