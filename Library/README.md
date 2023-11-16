# Neural Network Project

This is a simple implementation of a neural network in Python, focusing on dense layers and activation functions. The neural network is designed to handle regression tasks, and it includes a mean squared error (MSE) loss function for training.

## Layers

### Layer Class
The base class representing a generic neural network layer. It includes methods for forward and backward propagation.

```python
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