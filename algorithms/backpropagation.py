import math
import random
from shared import (
    get_csv_dataset, dot_product, dataset_minmax, normalize_dataset,
    denormalize_col
)


def init_network(n_inputs, n_hidden, n_outputs):
    random.seed(40)
    hidden_layer = [{
        'weights': [ random.random() for i in range(n_inputs + 1) ]
    } for i in range(n_hidden)]
    output_layer = [{
        'weights': [ random.random() for i in range(n_hidden + 1) ]
    } for i in range(n_outputs)]
    

    return [ hidden_layer, output_layer ]


def net_input(inputs, weights):
    """ Activation function """
    bias = weights[0]
    return bias + dot_product(inputs, weights[1:])

def squash(activation):
    """ Transfer function converts outputs into a smaller range
        Implementation uses the sigmoid function"""
    return 1.0 / (1.0 + math.exp(-activation))

def process_layer(layer, inputs):
    for neuron in layer:
        activation = net_input(inputs, neuron['weights'])
        neuron['output'] = squash(activation)
        yield neuron['output']

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        inputs = list(process_layer(layer, inputs))
    return inputs

def derivative_of_sigmoid(output):
    return output * (1.0 - output)

def backward_propagation(network, targets):
    # Loop through layers from output to hidden
    num_layers = len(network)
    for i in reversed(range(num_layers)):
        layer = network[i]
        errors = []
        if i != num_layers - 1:
            # Runs if not output layer
            for j in range(len(layer)):
                error = sum(neuron['weights'][j] * neuron['delta'] for neuron in network[i + 1])
                errors.append(error)
        else:
            for neuron, target in zip(layer, targets):
                errors.append(target - neuron['output'])
        # Get diff between error and output
        for neuron, error in zip(layer, errors):
            neuron['delta'] = error * derivative_of_sigmoid(neuron['output'])

def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        # inputs are outputs of next layer if first layer else data from the row in dataset
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            neuron['weights'][1:] = [weight + (learning_rate * neuron['delta'] * input_) for input_, weight in zip(inputs, neuron['weights'][1:])]
            neuron['weights'][0] += learning_rate * neuron['delta']

def preserve_weights(network):
    return [[
        [w for w in neuron['weights']] for neuron in layer
    ] for layer in network ]

def revert_weights(network, old_weights):
    for layer, weight_layer in zip(network, old_weights):
        for neuron, w_neuron in zip(layer, weight_layer):
            neuron['weights'] = w_neuron

def train_network(network, train_data, learning_rate, n_epoch, n_outputs):
    # n_epoch is number of times network is trained
    errors = []
    error_rates = []
    error_diff = 0.00001
    min_error_rate = 1.0e30
    for _epoch in range(n_epoch):
        total_error = 0
        random.shuffle(train_data)
        targets = [row[-1] for row in train_data]
        for row in train_data:
            outputs = forward_propagate(network, row)
            # Expected class using array indexes
            total_error += sum((target - output)**2 for target, output in zip(targets, outputs))
            backward_propagation(network, targets)
            update_weights(network, row, learning_rate)
        errors.append(total_error)
        error_rate = (1 / len(errors)) * total_error
        error_rates.append(error_rate)
        if min_error_rate > error_rate:
            min_error_rate = error_rate
        if total_error < error_diff or errors and errors[-1] <= total_error - error_diff:
            break
    return errors, error_rates


def get_closest(targets, normalized_targets, val):
    closest_val = min((abs(val - label), label) for label in targets)[1]
    # closest_index = list(normalized_targets).index(closest_val)
    # return list(targets)[closest_index]
    return closest_val

def predict(network, row, classes=None, minmax=[1,1]):
    expected = denormalize_col(row[-1], minmax[-1])
    outputs = forward_propagate(network, row)
    # print(expected, outputs)
    # print(expected, outputs, derivative_of_sigmoid(outputs[0]))
    denormalized_prediction = denormalize_col(outputs[0], minmax[-1])
    return expected, get_closest(classes, [], denormalized_prediction)
    # output_index = outputs.index(max(outputs))
    # return expected, output_index
    # return expected, classes[output_index] if classes else output_index


def main():
    dataset = [
        [2.7810836,2.550537003,0.0],
        [1.465489372,2.362125076,0.0],
        [3.396561688,4.400293529,0.0],
        [1.38807019,1.850220317,0.0],
        [3.06407232,3.005305973,0.0],
        [7.627531214,2.759262235,1.0],
        [5.332441248,2.088626775,1.0],
        [6.922596716,1.77106367,1.0],
        [8.675418651,-0.242068655,1.0],
        [7.673756466,3.508563011,1.0]
    ]
    learning_rate = 0.005
    n_epochs = 500
    # dataset = get_csv_dataset('wheat-seeds')
    # learning_rate = 0.05
    # n_epochs = 1000
    classes = list(set([row[-1] for row in dataset]))
    n_inputs = len(dataset[0]) - 1
    n_outputs = 1
    n_hidden = 5
    # n_epochs = 1500
    network = init_network(n_inputs, n_hidden, n_outputs)
    minmax = dataset_minmax(dataset)
    dataset = normalize_dataset(dataset, minmax)
    _errors, error_rates = train_network(network, dataset[:], learning_rate, n_epochs, n_outputs)
    predictions = []
    correct = {}
    for row in dataset:
        expected, actual = predict(network, row, classes, minmax=minmax)
        if expected == actual:
            correct[actual] = 1 + correct.get(actual, 0)
        predictions.append(f"Expected: {expected} Output: {actual}\n")
        print(f"Expected: {expected} Output: {actual}")
    
    print(correct)
    # import matplotlib.pyplot as plt
    # plt.plot(error_rates)
    # plt.show()
    
if __name__ == "__main__":
    main()