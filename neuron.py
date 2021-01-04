from . import weight
import random
from math import exp

LAMBDA = 0.3
ETA = 1.5
ALPHA = 0.8


class Neuron():
    def __init__(self, value=0.0, activation_val=0.0):
        self.value = value
        self.activation_val = activation_val
        self.connections = {}

    def activate(self):
        if self.value > 200:
            self.activation_val = 1.0
        elif self.value < -200:
            self.activation_val = 0.0
        else:
            self.activation_val = 1 / (1 + exp(LAMBDA * self.value * -1))

    def connect_neurons(self, neurons):
        # Reinitialize connections to empty dict
        self.connections = {}
        for neuron in neurons:
            self.connections[neuron] = weight.Weight(random.uniform(-1, 1))

    def __str__(self):
        result = "Val: {:.2f}, Act: {:.2f}, Conns: [".format(
            self.value, self.activation_val)
        for con in self.connections:
            result += "{} ".format(self.connections[con])
        result += "]"
        return result

    def feedforward(self):
        for n in self.connections:
            n.value += self.activation_val * self.connections[n].value

    def update_weights(self, gradients):
        for n, g in zip(self.connections, gradients):
            delta_w = ETA * self.activation_val * g + \
                ALPHA * self.connections[n].previous_delta
            self.connections[n].add_delta(delta_w)

    def calculate_output_gradient(self, error):
        return LAMBDA * self.activation_val * (1 - self.activation_val) * error

    def calculate_gradient(self, previous_gradients):
        result = 0.0
        for n, pg in zip(self.connections, previous_gradients):
            result += self.connections[n].previous_val * pg
        return (result * LAMBDA * self.activation_val * (1 - self.activation_val))
