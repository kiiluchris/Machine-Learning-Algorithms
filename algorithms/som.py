"""
create n x n map with random node vector values
loop while s < StepsMax times
  compute what a "close" node means, based on s
  compute a learn rate, based on s
  pick a random data item
  determine the map node closest to data item (BMU)
  for-each node close to the BMU
    adjust node vector values towards data item
end-loop
"""
import json
import math
import random

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from shared import (
    ManhattanDist, 
    get_csv_dataset,
    most_common
)


def EuclidianDist(v1, v2):
  return np.linalg.norm(v1 - v2)

def closest_node(row, map_, rows, cols):
    res = (0, 0)
    smallest_dist = 1.0e30
    for r in range(rows):
        for c in range(cols):
            dist = EuclidianDist(map_[r][c], row)
            if dist < smallest_dist:
                smallest_dist = dist
                res = (r, c)

    return res
    

def update_map(row, map_, rows, cols, bmu_indices, current_learning_rate, current_radius):
    for r in range(rows):
        for c in range(cols):
            if ManhattanDist(bmu_indices, [r,c]) < current_radius:
                map_[r][c] += current_learning_rate * (row - map_[r][c])


def construct_som(data_x, rows, cols, max_learning_rate, n_epochs, num_inputs):
    max_radius = rows + cols
    # num_inputs = len(input_cols)
    # map_ = create_som(rows, cols, num_inputs)
    map_ = np.random.random_sample(size=(rows, cols, num_inputs))
    for i in range(n_epochs):
        # Print if iteration is a multiple of 10
        if i % (n_epochs / 10) == 0:
            print(f"{i} epochs done", flush=True)

        # Decay
        pct_left = 1.0 - (i / n_epochs)
        current_radius = int(pct_left * max_radius)
        current_learning_rate = pct_left * max_learning_rate

        random_row_index = np.random.randint(len(data_x))
        row = data_x[random_row_index]
        bmu_indices = closest_node(row, map_, rows, cols)
        update_map(row, map_, rows, cols, bmu_indices, current_learning_rate, current_radius)

    return map_


def construct_u_matrix(map_, rows, cols):
    """ Create a u matrix out of the self organized map """
    u_matrix = np.zeros(shape=(rows,cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            vector_ = map_[r][c]
            total_distance = 0.0
            counter = 0.0
            # if statement order: top, bottom, left, right
            if r - 1 >= 0:
                total_distance += EuclidianDist(vector_, map_[r - 1][c])
                counter += 1
            if r + 1 <= rows - 1:
                total_distance += EuclidianDist(vector_, map_[r + 1][c])
                counter += 1
            if c - 1 >= 0:
                total_distance += EuclidianDist(vector_, map_[r][c - 1])
                counter += 1
            if c + 1 <= cols - 1:
                total_distance += EuclidianDist(vector_, map_[r][c + 1])
                counter += 1

            u_matrix[r][c] = total_distance / counter

    return u_matrix


def display_umatrix(u_matrix):
    plt.figure(1)
    plt.imshow(u_matrix, cmap='gray')

def dimensionality_reduction_visualization(map_, u_matrix, data_x, data_y, rows, cols):
    mapping = np.empty(shape=(rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            mapping[r][c] = []
    for i in range(len(data_x)):
        bmu_row, bmu_col = closest_node(data_x[i], map_, rows, cols)
        mapping[bmu_row][bmu_col].append(data_y[i])

    label_map = np.zeros(shape=(rows,cols), dtype=np.int)
    for i in range(rows):
        for j in range(cols):
            label_map[i][j] = most_common(mapping[i][j])

    plt.figure(2)
    plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4))
    plt.colorbar()

def load_dataset(dataset_name, input_indices):
    file_path = Path(__file__).parent / f"./datasets/{dataset_name}.csv"
    num_inputs = len(input_indices)
    data_x = np.loadtxt(file_path.as_posix(), delimiter=",", usecols=input_indices,
        dtype=np.float64)
    data_y = np.loadtxt(file_path.as_posix(), delimiter=",", usecols=[num_inputs],
        dtype=np.int)

    return data_x, data_y, num_inputs

def main():
    np.random.seed(1)
    # Define som dimensions
    rows, cols = 30, 30
    # data_x, data_y, num_inputs = load_dataset('iris-data', range(4))
    data_x, data_y, num_inputs = load_dataset('wheat-seeds', range(7))
    map_ = construct_som(
        data_x=data_x,
        rows=rows,
        cols=cols,
        max_learning_rate=0.5,
        n_epochs=1000,
        # n_epochs=5000,
        num_inputs=num_inputs
    )
    u_matrix = construct_u_matrix(map_, rows, cols)
    display_umatrix(u_matrix)
    dimensionality_reduction_visualization(map_, u_matrix, data_x, data_y, rows, cols)
    plt.show()

if __name__ == "__main__":
    main()
