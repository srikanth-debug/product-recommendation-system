def precision_at_k(actual, predicted, k=5):
    actual = set(actual)
    predicted = predicted[:k]
    return len(actual.intersection(predicted)) / k
