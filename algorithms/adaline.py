import json
import operator
import random

from shared import get_csv_dataset, normalize_dataset, dataset_minmax, sigmoid, denormalize_col, derivative_of_sigmoid
from shared import dot_product


def predict(inputs, weights):
    """ weights[0] == bias
        activation = bias + sum(x_i * w_i)"""
    bias = weights[0]
    squashed_inputs= inputs
    squashed_inputs = [sigmoid(input_) for input_ in inputs]
    activation = dot_product(squashed_inputs, weights[1:]) + bias
    return activation


def train_weights(train, targets, learning_rate, n_epoch, label_index=-1):
    weights = [random.random() for _ in train[0]]
    min_sq_error = 1.0e30
    for _epoch in range(n_epoch):
        errors = []
        random.shuffle(train)
        # old_bias = weights[0]
        # old_weights = weights[1:]
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            errors.append(error)
            weights[0] += learning_rate * error
            weights[1:] = [ 
                weight + learning_rate * error * input_
                for weight, error, input_ in zip(weights[1:], errors, row) 
            ]
        cost = sum(error ** 2 for error in errors)
        if min_sq_error > cost:
            min_sq_error = cost
        # else:
        #     weights[0] = old_bias
        #     weights[1:] = old_weights
        # elif min_sq_error < 0.0001:
        else:
            return weights
        # costs.append(cost)

    return weights
    
def get_closest(targets, normalized_targets, val):
    closest_val = min((abs(val - label), label) for label in targets)[1]
    # closest_index = list(normalized_targets).index(closest_val)
    # return list(targets)[closest_index]
    return closest_val


dataset = [
    [2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]
]

learning_rate = 0.05
# dataset = get_csv_dataset('wheat-seeds')
# learning_rate = 0.07
n_epoch = 600
label_index = -1
o_targets = list(map(operator.itemgetter(label_index), dataset))
minmax = dataset_minmax(dataset)
dataset = normalize_dataset(dataset, minmax)
targets = list(map(operator.itemgetter(label_index), dataset))
random.seed(1)
weights = train_weights(dataset[:], targets, learning_rate, n_epoch)
labels = set(o_targets)
norm_labels = set(targets)
res = {
    "predictions": [],
    "counters": {}
}
for i, row in enumerate(dataset):
    prediction = predict(row, weights)
    denormalized_prediction = prediction
    denormalized_prediction = denormalize_col(prediction, minmax[label_index])
    predicted_class = get_closest(labels, norm_labels, denormalized_prediction)
    target = o_targets[i]
    print(f"Expected: {target} Output: {predicted_class}\n")
    res["predictions"].append({
        "val": predicted_class,
        "expected": target,
    })
    if predicted_class == target:
        res["counters"][target] = res["counters"].get(target, 0) + 1
print(res["counters"])