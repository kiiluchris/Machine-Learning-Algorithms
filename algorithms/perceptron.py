from .shared import dot_product


def gradient_descent(dataset, weights):
    return (float(weights[0] + dot_product(row, weights[1:]) >= 0.0) for row in dataset )
    
activation_function = gradient_descent

def predict(inputs, weights):
    # weights[0] == bias
    # activation = bias + sum(x_i * w_i)
    bias = weights[0]
    activation = dot_product(inputs, weights[1:]) + bias
    return float(activation >= 0.0)

def train_weights(train, learning_rate, n_epoch):
    weights = [0.0] * len(train[0])
    for epoch in range(n_epoch):
        total_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            total_error += error
            # New bias
            weights[0] = weights[0] + learning_rate * error
            weights[1:] = [ weight + learning_rate * error * input_ for weight, input_ in zip(weights[1:], row)]

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, total_error))
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