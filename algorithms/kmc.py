import operator
from .shared import EuclidianDist, transpose

from functools import reduce, partial

real_map = map
def map(*args):
    return list(real_map(*args))

def groupsHaveChanged(oldGroups, groups):
    if not oldGroups and not groups: return True
    for old_group, group in zip(oldGroups, groups):
        if old_group != group:
            return True
    return False

def find_groups(distances):
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
    groupTotals = (sum(v) for v in vals)
    return map(lambda x:  x / k, groupTotals)

def KMC(data, k):
    oldGroups = []
    groups = []
    counter = 10
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
