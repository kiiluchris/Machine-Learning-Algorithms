import operator

from .shared import dot_product


def predict(inputs, weights):
    # weights[0] == bias
    # activation = bias + sum(x_i * w_i)
    bias = weights[0]
    activation = dot_product(inputs, weights[1:]) + bias
    return float(activation >= 0.0)

def train_weights(train, learning_rate, n_epoch):
    weights = [0.0] * len(train[0])
    targets = list(map(operator.itemgetter(2), train))
    costs = []
    for epoch in range(n_epoch):
        predictions = [predict(inputs, weights) for inputs in train]
        errors = [target - prediction for prediction, target in zip(predictions, targets)]
        weights[0] += learning_rate * sum(errors)
        inputs_T = zip(*train[:-1])
        weight_deviations = [ dot_product(input_, errors) for input_ in inputs_T]
        weights[1:] = [ weight + learning_rate * error for weight, error in zip(weights[1:], weight_deviations) ]
        cost = sum(error ** 2 for error in errors)
        costs.append(cost)

    return weights

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

learning_rate = 0.1
n_epoch = 200
weights = train_weights(dataset, learning_rate, n_epoch)

for row in dataset:
    print(predict(row, weights))