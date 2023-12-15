import numpy as np
import json

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        #TODO: return input
        pass

    def backward(self, output_gradient, learning_rate):
        #TODO: update parameters and return input gradient
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
    

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


#must have a 'saves.json' file
#saves all weights and biases of a layer array(network) to a saves.json file
def save(layers, filename='saves.json'):
    network_data = []

    for layer in layers:
        layer_data = {}
        if isinstance(layer, Dense):
            layer_data["weights"] = layer.weights.tolist()
            layer_data["bias"] = layer.bias.tolist()
        elif isinstance(layer, Activation):
            layer_data["activation"] = layer.activation.__name__
            layer_data["activation_prime"] = layer.activation_prime.__name__
        
        network_data.append(layer_data)

    with open(filename, 'w') as file:
        json.dump({"network": network_data}, file, indent=2)
        
#loads weights and biases onto compatible layer array(network) from saves.json file
def load(network, filename='saves.json'):
    with open(filename, 'r') as file:
        data = json.load(file)

    for layer, layer_data in zip(network, data['network']):
        if isinstance(layer, Dense):
            layer.weights = np.array(layer_data["weights"])
            layer.bias = np.array(layer_data["bias"])
