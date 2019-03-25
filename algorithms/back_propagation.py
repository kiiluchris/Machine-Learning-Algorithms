from .shared import (
    sigmoid, get_csv_dataset, dot_product, calculate_net_input, derivative_of_sigmoid
)
import random

squash = sigmoid

def gen_weights(n_prev, n_current):
    return [{
        "weights": random.random() for _ in range(n_prev + 1)
    } for n in range(n_current)]


def create_network(n_inputs, n_hidden, n_outputs):
    random.seed(1)
    hidden_layer = gen_weights(n_inputs, n_hidden)
    output_layer = gen_weights(n_hidden, n_outputs)

    return [ hidden_layer, output_layer ]

def normalized_inputs(inputs):
    return [sigmoid(input_) for input_ in inputs[1:]]

def process_layer(layer, inputs):
    for neuron in layer:
        net_input = calculate_net_input(normalized_inputs(inputs), weights)
        neuron['output'] = squash(net_input)
        yield neuron['output']

def forward_pass(network, row):
    inputs = row
    for layer in network:
        inputs = list(process_layer(layer, inputs))
    return inputs

def get_total_error(dataset, outputs):
    total_error = 0.0
    for i in len(outputs):
        total_error += ((dataset[i][-1] - outputs[i]) ** 2) / 2
    return total_error

def backward_pass(network, dataset):
    layer['errors'] = get_total_error(dataset, network[i - 1]['outputs'])
    last_index = len(network) - 1
    target = [row[-1] for row in dataset]
    for i in reversed(last_index + 1):
        layer = network[i]
        errors = []
        # for last later
        if i != last_index:
            for j in range(len(layer)):
                error = sum(neuron['weights'][j] * neuron['delta'] for neuron in network[i + 1])
                errors.append(error)
        else:
            for neuron, target in zip(layer, target):
                errors.append(neuron['output'] - target)
          
        for neuron, error in zip(layer, errors):
            neuron['delta'] = error * derivative_of_sigmoid(neuron['output'])      
        
    return

def main():
    dataset =msf get_csv_dataset('back-p')
    network = create_network(2, 2, 2)
    for row in dataset[:10]:
        outputs = forward_pass(network, row)




if __name__ == "__main__":
    main()