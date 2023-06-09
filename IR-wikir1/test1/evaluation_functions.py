from collections import defaultdict

import ir_datasets

dataset = ir_datasets.load("wikir/en1k/training")


def calculate_precision_at_k(retrieval_results, qrels_dict, k=10):
    precisions = {}
    for query_id, retrieved_docs in retrieval_results.items():
        print("query_id")
        print(query_id)
        # print(qrels_dict)
        relevant_docs = qrels_dict[query_id]
        print("relevant_docs")
        print(relevant_docs)
        print("retrieval_results")
        print(retrieval_results)
        retrieved_docs_at_k = retrieved_docs[:k]  # Consider only top-k retrieved documents
        print("retrieved_docs_at_k")
        print(retrieved_docs_at_k)
        retrieved_relevant_docs = 0

        for doc_id in retrieved_docs_at_k:
            if doc_id in relevant_docs:
                retrieved_relevant_docs += 1

        precision = retrieved_relevant_docs / k
        precisions[query_id] = precision

    return precisions


def calculate_recall(retrieval_results, qrels_dict):
    recalls = {}
    for query_id, retrieved_docs in retrieval_results.items():
        relevant_docs = qrels_dict.get(query_id, [])
        total_relevant_docs = len(relevant_docs)
        retrieved_relevant_docs = 0

        for doc_id in retrieved_docs:
            if doc_id in relevant_docs:
                retrieved_relevant_docs += 1

        recall = retrieved_relevant_docs / total_relevant_docs if total_relevant_docs > 0 else 0
        recalls[query_id] = recall

    return recalls


def calculate_average_precision(retrieval_results, qrels_dict):
    average_precisions = []
    # Calculate Average Precision for each query
    for query_id, retrieved_docs in retrieval_results.items():
        relevant_docs = qrels_dict.get(query_id, [])
        precision_sum = 0.0
        retrieved_relevant_docs = 0

        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                retrieved_relevant_docs += 1
                precision = retrieved_relevant_docs / (i + 1)
                precision_sum += precision

        average_precision = precision_sum / len(relevant_docs) if relevant_docs else 0.0
        average_precisions.append(average_precision)

    # Calculate Mean Average Precision (MAP)
    average_score = sum(average_precisions) / len(average_precisions)
    return average_score


def calculate_mrr(retrieval_results, qrels_dict):
    reciprocal_ranks = []

    for query_id, retrieved_docs in retrieval_results.items():
        relevant_docs = qrels_dict.get(query_id, [])  # Replace with your relevant documents retrieval method

        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                reciprocal_ranks.append(1 / rank)
                break

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return mrr


def calculate_evaluation(retrieval_results):
    qrels_dict = defaultdict(list)
    for qrel in dataset.qrels_iter():
        qrels_dict[qrel.query_id].append(qrel.doc_id)

    precision = calculate_precision_at_k(retrieval_results, qrels_dict)
    print("Precision@10 : ", precision)
    recall = calculate_recall(retrieval_results, qrels_dict)
    print("Recall : ", recall)
    average = calculate_average_precision(retrieval_results, qrels_dict)
    print("MAP : ", average)
    reciprocal = calculate_mrr(retrieval_results, qrels_dict)
    print("MRR : ", reciprocal)
