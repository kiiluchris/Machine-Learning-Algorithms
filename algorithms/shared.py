import csv
import math
import os
from math import sqrt
from operator import itemgetter
from functools import reduce, partial
from pathlib import Path

DATASET_DIR = Path(__file__).parent.joinpath('./datasets')

def transpose(*arrs): 
    return zip(*arrs)


def most_common(xs, default=-1):
    counts = {}
    for x in xs:
        counts[x] = 1 + counts.get(x, 0)
    return max(counts.items(), key=itemgetter(1), default=[default])[0]

def EuclidianDist(*points):
    squaredDist = reduce(lambda acc, p : acc + pow(p[0] - p[1], 2), points, 0)
    return sqrt(squaredDist)

def ManhattanDist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def dot_product(inputs, weights):
    return sum([i * w for i, w in zip(inputs, weights)])
    
def calculate_net_input(inputs, weights):
    """ Activation function """
    bias = weights[0]
    return bias + dot_product(inputs, weights[:-1])

def get_csv_dataset(path, label_index=-1):
    dataset = get_csv_dataset_text(path)
    return [[
            float(val) if val else 0.0 for val in row
    ] for row in dataset]

def get_csv_text(path):
    with open(path, 'r', newline='') as csvfile:
        my_reader = csv.reader(csvfile, delimiter=',')
        return [[
            col.strip() for col in row if col.strip()
        ] for row in my_reader]

def get_csv_dataset_text(path):
    return get_csv_text(str(DATASET_DIR.joinpath(f'{path}.csv')))

def current_dir(file_): 
    return os.path.dirname(file_)

def dataset_path(file_, dataset):
    dir_ = current_dir(file_)
    return os.path.join(dir_, '', 'datasets', f'{dataset}.csv')


def wheat_seeds_csv(file_):
    return get_csv_dataset(dataset_path(file_, 'wheat-seeds'))


def derivative_of_sigmoid(output):
    return output * (1.0 - output)


def sigmoid(value):
    """ Transfer function converts outputs into a smaller range
        Implementation uses the sigmoid function"""
    return 1.0 / (1.0 + math.exp(-value))


def dataset_minmax(dataset):
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats
   
def denormalize_row(row, minmax):
    return [
        denormalize_col(row[i], minmax[i])
    for i, _ in enumerate(row) ]

def denormalize_col(col, minmax_col):
    min_, max_ = minmax_col
    return (col * (max_ - min_)) + min_

def normalize_col(col, minmax_col):
    min_, max_ = minmax_col
    return (col - min_) / (max_ - min_)

def normalize_row(row, minmax):
    return [
        normalize_col(row[i], minmax[i])
        for i, col in enumerate(row)
    ]

def normalize_dataset(dataset, minmax):
    return [
        normalize_row(row, minmax)
        for row in dataset
    ]