import numpy as np

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # init weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # calc ouput vals from inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backwar(self, dvalues):
        # gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
