import math
from inverted_index import create_inverted_index
def calculate_tf(document):
    tf = {}
    terms = document.split()
    term_count = len(terms)
    for term in terms:
        tf[term] = terms.count(term) / term_count
    return tf


def calculate_idf(corpus):
    idf = {}
    n_docs = len(corpus)
    inverted_index = create_inverted_index(corpus)
    for term, doc_ids in inverted_index.items():
        idf[term] = math.log(n_docs / len(doc_ids))
    return idf

# print(calculate_idf(preprocessed_docs))

def calculate_tfidf(document, corpus):
    tfidf = {}
    tf = calculate_tf(document)
    idf = calculate_idf(corpus)
    for term in tf:
        tfidf[term] = tf[term] * idf[term]
    return tfidf
