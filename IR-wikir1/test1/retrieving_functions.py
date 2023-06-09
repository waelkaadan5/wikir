from collections import defaultdict
import colorama
from colorama import Fore
import pandas as pd
import numpy as np
import string
import re
import nltk
import math
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_function import preprocess
from sklearn.cluster import KMeans

import ir_datasets

from test1.evaluation_functions import calculate_precision_at_k, calculate_recall, calculate_evaluation

dataset = ir_datasets.load("wikir/en1k/training")

# def get_query_(text):
#     with open(r"C:\Users\hp\Desktop\test\queries.txt", 'r') as file:
#         for line in file:
#             query = line.rstrip('\n').split('\t')
#             if query[1] == text:
#                 return query[0]


for q in dataset.queries_iter():
    print(q)
def get_query_id(text):
    for q in dataset.queries_iter():
        if q.text == text:
            return q.query_id


# preprocessed_docs = pickle.load(open("preprocessed_docs.pickle", "rb"))
# preprocessed_queries = pickle.load(open("preprocessed_queries.pickle", "rb"))

# create_inverted_index(preprocessed_docs)

# inverted_index = pickle.load(open("tfidf[docs].pickle", "rb"))
# print(inverted_index)

# print(calculate_idf(preprocessed_docs))

# print(calculate_tfidf(preprocessed_docs['2020338_1'], preprocessed_docs))

# documents = list(preprocessed_docs.values())
# queries = list(preprocessed_queries.values())
query = "what is direct borrowing, phonological borrowing, linguistic borrowing, cultural borrowing,intimate borrowing?"

def get_top_related_docs(my_query, num_docs=10):
    docs_dict = defaultdict(list)
    for doc in dataset.docs_iter():
        docs_dict[doc.doc_id] = doc.text
    preprocessed_docs = pickle.load(open("preprocessed_docs.pickle", "rb"))
    inverted_index = pickle.load(open("tfidf[docs].pickle", "rb"))
    my_preprocessed_query = preprocess(my_query)
    query_terms = my_preprocessed_query.split()
    print(query_terms)
    related_doc_ids = []
    for term in query_terms:
        print(term)
        if term in inverted_index:
            # print(term)
            # print(inverted_index[term])
            related_doc_ids.extend(inverted_index[term])
    related_doc_ids = list(set(related_doc_ids))
    related_doc_text = [preprocessed_docs[doc_id] for doc_id in related_doc_ids]
    vectorizer = TfidfVectorizer()
    docs_tfidf = vectorizer.fit_transform(related_doc_text)
    query_tfidf = vectorizer.transform([my_preprocessed_query])
    cosine_similarities = cosine_similarity(docs_tfidf, query_tfidf)
    queryId = get_query_id(my_query)
    print(queryId)
    top_doc_indices = cosine_similarities.argsort(axis=0)[-num_docs:][::-1].flatten()
    top_scores = cosine_similarities[top_doc_indices].flatten()
    top_docs = []
    top_docs_text = []

    for i, doc_index in enumerate(top_doc_indices):
        doc_id = related_doc_ids[doc_index]
        top_docs.append(doc_id)
        top_docs_text.append(docs_dict[doc_id])

    retrieval_results = {}
    retrieval_results[queryId] = top_docs
    for indx, itm in enumerate(top_docs):
        print("Document: ", top_docs[indx], ", Similarity:", top_scores[indx])
    print(retrieval_results)
    calculate_evaluation(retrieval_results)
    return top_docs, top_docs_text, top_scores.tolist()


def get_top_related_docs_clustered(my_query, num_clusters=30, top_k=10):
    docs_dict = defaultdict(list)
    for doc in dataset.docs_iter():
        docs_dict[doc.doc_id] = doc.text
    preprocessed_docs = pickle.load(open("preprocessed_docs.pickle", "rb"))
    inverted_index = pickle.load(open("tfidf[docs].pickle", "rb"))
    my_preprocessed_query = preprocess(my_query)
    query_terms = my_preprocessed_query.split()
    related_doc_ids = []
    for term in query_terms:
        if term in inverted_index:
            related_doc_ids.extend(inverted_index[term])
    related_doc_ids = list(set(related_doc_ids))
    related_doc_text = [preprocessed_docs[doc_id] for doc_id in related_doc_ids]

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(related_doc_text)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(doc_vectors)

    # Assuming you have preprocessed the query and transformed it into a feature vector called 'query_vector'
    query_vector = vectorizer.transform([preprocess(my_query)])
    queryId = get_query_id(my_query)

    # Assign the query to a cluster
    query_cluster_label = kmeans.predict(query_vector)[0]
    np.set_printoptions(threshold=np.inf)

    # Retrieve relevant documents from the assigned cluster
    relevant_cluster_indices = [i for i, label in enumerate(cluster_labels) if label == query_cluster_label]
    relevant_documents_ids = [related_doc_ids[i] for i in relevant_cluster_indices]
    relevant_documents_text = [related_doc_text[i] for i in relevant_cluster_indices]

    # Calculate similarity between query and documents in the assigned cluster
    similarities = cosine_similarity(query_vector, doc_vectors[relevant_cluster_indices])
    similarities = similarities.flatten()

    # Rank the relevant documents based on similarity
    top_doc_indices = similarities.argsort()[::-1][:top_k]
    top_similarities = [similarities[i] for i in top_doc_indices]
    top_documents_ids = [relevant_documents_ids[i] for i in top_doc_indices]
    top_documents_text = [docs_dict[id] for id in top_documents_ids]

    retrieval_results = {}
    retrieval_results[queryId] = top_documents_ids

    for indx, itm in enumerate(top_documents_ids):
        print("Document: ", top_documents_ids[indx], ", Similarity:", top_similarities[indx])

    calculate_evaluation(retrieval_results)

    return top_documents_ids, top_documents_text, top_similarities


# print(get_top_related_docs_clustered(query))

# vectorizer = TfidfVectorizer()
# corpus_tfidf_matrix = vectorizer.fit_transform(documents)
# df = pd.DataFrame(corpus_tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=preprocessed_docs.keys())
# # print(df)
#
# query_tfidf = vectorizer.transform(queries)
#
# cosine_similarities = cosine_similarity(corpus_tfidf_matrix,query_tfidf)
#
# num_top_documents = 3
#
# retrieval_results = {}
#
# for i, query in enumerate(dataset.queries_iter()[:20]):
#     print(f"Query: {query.text}")
#     top_documents_indices = cosine_similarities[i].argsort()[-num_top_documents:][::-1]
#     for doc_index in top_documents_indices:
#         similarity_score = cosine_similarities[i, doc_index]
#         doc_id = list(preprocessed_docs.keys())[doc_index]
#         print(f"  Document {doc_id}: {similarity_score}")
#         # print(preprocessed_docs[doc_id])
#         retrieval_results = {}
#
# # Retrieve top 10 documents for each query
# for i, query in enumerate(dataset.queries_iter()[:20]):
#     query_id = query.query_id
#     retrieval_results[query_id] = []
#
#     # Get cosine similarity scores for the query
#     similarity_scores = cosine_similarities[i]
#
#     # Sort document indices based on similarity scores
#     top_indices = similarity_scores.argsort()[::-1][:num_top_documents]
#
#     # Retrieve document IDs using the sorted indices
#     top_documents = [list(preprocessed_docs.keys())[index] for index in top_indices]
#
#     # Add retrieved documents to the retrieval_results dictionary
#     retrieval_results[query_id] = top_documents
#
# # Print retrieval_results dictionary
# print(retrieval_results)
#
# print(calculate_precision_at_k(retrieval_results,10))
#
# print(calculate_recall(retrieval_results))
