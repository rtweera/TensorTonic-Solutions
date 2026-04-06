def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    precision_at_k = len(set(recommended[:k]) & set(relevant))/k
    recall_at_k = len(set(recommended[:k]) & set(relevant))/len(set(relevant))
    return [precision_at_k, recall_at_k]