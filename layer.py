from . import neuron


class Layer():
    def __init__(self, amount, name, is_input=False):
        self.neurons = []
        self.name = name
        self.is_input = is_input
        self.has_bias = False
        for _ in range(amount):
            n = neuron.Neuron()
            self.neurons.append(n)

    def connect_layer(self, other, has_bias=True):
        for n in self.neurons:
            n.connect_neurons(other.neurons)
        if not has_bias:
            return
        self.has_bias = True
        bias = neuron.Neuron(value=1000.0, activation_val=1.0)
        bias.connect_neurons(other.neurons)
        self.neurons.append(bias)

    def __str__(self):
        result = self.name + ":\n"
        for i, n in enumerate(self.neurons):
            if i+1 == len(self.neurons) and self.has_bias:
                result += "Bias: {}\n".format(str(n))
            else:
                result += "Neuron #{}: {}\n".format(i, str(n))
        return result

    def add_inputs(self, input):
        for i in range(len(input)):
            self.neurons[i].value = input[i]

    def feedforward(self):
        for i, n in enumerate(self.neurons):
            n.activate()
            # If layer is input layer, pass the original value, not the
            # activation value
            if self.is_input and not(self.has_bias and i+1 == len(self.neurons)):
                n.activation_val = n.value
            n.feedforward()

    def get_values(self):
        return [n.activation_val for n in self.neurons]

    def update_weights(self, gradients):
        for n in self.neurons:
            n.update_weights(gradients)

    def calculate_output_gradients(self, errors):
        gradients = []
        for n, e in zip(self.neurons, errors):
            gradients.append(n.calculate_output_gradient(e))
        return gradients

    def calculate_gradients(self, previous_gradients):
        gradients = []
        for n in self.neurons:
            gradients.append(n.calculate_gradient(previous_gradients))
        return gradients

    def zero_neurons(self):
        for i, n in enumerate(self.neurons):
            if i+1 == len(self.neurons) and self.has_bias:
                break
            n.value = 0.0
            n.activation_val = 0.0
