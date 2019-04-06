import operator
from functools import reduce, partial
from shared import EuclidianDist, transpose
from shared import get_csv_dataset

def data_to_distance_with_label(query, label_index, value):
    data, label = [val for i, val in enumerate(value) if i != label_index ], value[label_index]
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

def KNN(dataset, query, k, label_index=-1):
    # Create partial function with query preadded as first parameter
    dist_mapper = partial(data_to_distance_with_label, query, label_index)
    # Calculate distance from query return as List of (distance, label)
    distancesWithLabels = map(dist_mapper, dataset)
    # Sort distances is ascending order and get k nearest neighbours
    sortedDistancesWithLabel = sorted(distancesWithLabels, key=operator.itemgetter(0))[:k]
    # Count labels in the k neighbours as in vote
    counts = reduce(count_labels, sortedDistancesWithLabel, {})

    # Return label of the highest vote
    return reduce(max_label_count, counts.items())

if __name__ == "__main__":
    dataset = get_csv_dataset('wheat-seeds')
    # dataset = [
    #     [2,3,'lol'], 
    #     [4,5,'lol'],
    #     [1,3,'no'],
    #     [3,7,'hah'],
    #     [3,3,'hey']
    # ]
    query = [40,23]
    knn = KNN(dataset, query=query, k=8)
    print("Query:", query)
    print("Label:", knn[0])



