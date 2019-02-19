import operator
from functools import reduce, partial
from .shared import EuclidianDist, transpose

def data_to_distance_with_label(query, value):
    label, *data = value
    return EuclidianDist(*transpose(query, data)), label

def count_labels(acc, value):
    _, label = value
    return {
        **acc,
        label: acc.get(label, 0) + 1
    }

def max_label_count(prev, current):
    _, prev_count = prev
    _, current_count = current
    return current if current_count > prev_count else prev

def KNN(query, vals, k):
    dist_mapper = partial(data_to_distance_with_label, query)
    distancesWithLabels = map(dist_mapper, vals)
    sortedDistancesWithLabel = sorted(distancesWithLabels, key=operator.itemgetter(0))[:k]
    counts = reduce(count_labels, sortedDistancesWithLabel, {})
    
    return reduce(max_label_count, counts.items())


