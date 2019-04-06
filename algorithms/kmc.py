import operator
from shared import EuclidianDist, transpose, get_csv_dataset

from functools import reduce, partial

real_map = map
def map(*args):
    return list(real_map(*args))

def groupsHaveChanged(oldGroups, groups):
    """ Groups have changed if groups are both empty (just initialized) 
        or groups have any differing values
    """
    if not oldGroups and not groups: return True
    for old_group, group in zip(oldGroups, groups):
        if old_group != group:
            return True
    return False

def find_groups(distances):
    """ An element is in the group with the least minimum distance 
        A group is represented by a vector of zeros and a single one
    """
    minIndex = distances.index(min(distances))
    vals = [0] * len(distances)
    vals[minIndex] = 1
    return vals

def setup_grouper(data):
    def group_data_as_dict(acc, val):
        i, grouping = val
        groupIndex = grouping.index(1)
        return {
            **acc,
            groupIndex: [*acc.get(groupIndex, []), data[i]], 
        }
    return group_data_as_dict

def new_centroids(k, vals):
    """ New centroids generated as average of values in groups """
    groupTotals = (sum(v) for v in vals)
    return map(lambda x:  x / k, groupTotals)

def KMC(data, k):
    oldGroups = []
    groups = []
    counter = 100
    centroids = data[:k]
    group_data_as_dict = setup_grouper(data)
    centroid_mapper = partial(new_centroids, k)
    groupedData = {}
    while groupsHaveChanged(oldGroups, groups) and counter > 0:
        oldGroups = groups
        distances = map(lambda d:  map(lambda c:  EuclidianDist(*transpose(d, c)), centroids), data)
        groups = map(find_groups, distances)
        groupedDataObj = reduce(group_data_as_dict, enumerate(groups), {})
        groupedData = groupedDataObj.values()
        centroids = map(centroid_mapper, groupedData)
        counter -= 1
    
    return list(groupedData)


def main():
    dataset = [row[:-1] for row  in get_csv_dataset('wheat-seeds')]
    clusters = KMC(dataset, 5)
    print("Dataset length: ", len(dataset))
    print("Num of clusters (k):", len(clusters))
    print("Cluster sizes")
    for i, cluster in enumerate(clusters):
        print(f"     Cluster {i+1} length", len(cluster))


if __name__ == "__main__":
    main()