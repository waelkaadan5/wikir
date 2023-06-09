from test1.retrieving_functions import get_top_related_docs, get_top_related_docs_clustered

query = "How does a hold up alarm/silent alarm in a shop work?"
print(get_top_related_docs(query))
print(get_top_related_docs_clustered(query))
