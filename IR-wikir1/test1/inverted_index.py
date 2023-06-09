from collections import defaultdict
import nltk
import pickle
preprocessed_docs = pickle.load(open("preprocessed_docs.pickle", "rb"))
def create_inverted_index(corpus):
    inverted_index = defaultdict(list)
    for doc_id, doc_content in corpus.items():
        terms = nltk.word_tokenize(doc_content)
        for term in terms:
            inverted_index[term].append(doc_id)
    pickle.dump(dict(inverted_index), open("tfidf[docs]"+".pickle", "wb"))
    # return dict(inverted_index)

create_inverted_index(preprocessed_docs)