from functools import reduce


def transpose(*arrs): 
    return zip(*arrs)

def EuclidianDist(*points):
    squaredDist = reduce(lambda acc, p : acc + pow(p[0] - p[1], 2), points, 0)
    return squaredDist

def dot_product(inputs, weights):
    return sum([i * w for i, w in zip(inputs, weights)])
