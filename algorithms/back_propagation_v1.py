import random
from .shared import get_csv_dataset, dot_product
import math


def init_network(n_inputs, n_hidden, n_outputs):
    random.seed(1)
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
    return bias + dot_product(inputs, weights[:-1])

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

def train_network(network, train_data, learning_rate, n_epoch, n_outputs):
    # n_epoch is number of times network is trained
    targets = [row[-1] for row in train_data]
    for epoch in range(n_epoch):
        total_error = 0
        for row in train_data:
            outputs = forward_propagate(network, row)
            # Expected class using array indexes
            total_error += sum((target - output)**2 for target, output in zip(targets, outputs))
            backward_propagation(network, targets)
            update_weights(network, row, learning_rate)

        # print(f'epoch={epoch} learning_rate={learning_rate} total_error={total_error}')
        if total_error < 0.001:
            break



def predict(network, row, classes=None):
    outputs = forward_propagate(network, row)
    output_index = outputs.index(max(outputs))
    return classes[output_index] if classes else output_index


def dataset_minmax(dataset):
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats
   
def normalize_dataset(dataset, minmax):
    for r_i, row in enumerate(dataset):
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# dataset = [[2.7810836,2.550537003,0],
# [1.465489372,2.362125076,0],
# [3.396561688,4.400293529,0],
# [1.38807019,1.850220317,0],
# [3.06407232,3.005305973,0],
# [7.627531214,2.759262235,1],
# [5.332441248,2.088626775,1],
# [6.922596716,1.77106367,1],
# [8.675418651,-0.242068655,1],
# [7.673756466,3.508563011,1]]
def main():
    dataset = get_csv_dataset('wheat-seeds')

    classes = list(set([row[-1] for row in dataset]))
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(classes)
    n_hidden = 2
    learning_rate = 0.2
    n_epochs = 10000
    network = init_network(n_inputs, n_hidden, n_outputs)
    normalize_dataset(dataset, dataset_minmax(dataset))
    train_network(network, dataset, learning_rate, n_epochs, n_outputs)
    predictions = []
    for row in dataset:
        predictions.append(f"Expected: {row[-1]} Output: {predict(network, row, classes)}\n")
    import json
    with open('res.txt', 'w') as f:
        f.writelines(predictions)
    
if __name__ == "__main__":
    main()