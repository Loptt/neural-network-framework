import json
from . import layer


class NeuralNetwork():

    def __init__(self, name):
        self.name = name
        self.input_extremes = []
        self.output_extremes = []
        self.layers = []

    def add(self, layer, has_bias=True):
        if len(self.layers) > 0:
            self.layers[-1].connect_layer(layer, has_bias)
        else:
            # If there is no layer, this layer is input layer
            layer.is_input = True
        self.layers.append(layer)

    def __str__(self):
        result = "Network: {}\n".format(self.name)
        result += "==============================================\n"

        for layer in self.layers:
            result += "{}\n".format(str(layer))

        result += "==============================================\n"
        return result

    def feedforward(self, inputs):
        for layer in self.layers:
            layer.zero_neurons()
        input_layer = self.layers[0]
        input_layer.add_inputs(inputs)

        for layer in self.layers:
            layer.feedforward()

        output_layer = self.layers[-1]

        return output_layer.get_values()

    def train(self, inputs, outputs, x_val, y_val, epochs, stepped):
        history = [[], [], []]
        for i in range(epochs):
            for inp, expected in zip(inputs, outputs):
                output = self.feedforward(inp)
                self.backpropagate(output, expected, stepped)
                if stepped:
                    print(self)
                    input("ENTER")
            train_error = self.calculate_error(inputs, outputs)
            val_error = self.calculate_error(x_val, y_val)
            history[0].append(i+1)
            history[1].append(train_error)
            history[2].append(val_error)
            print("Epoch {}: Training error: {}, Val error: {}".format(
                i+1, train_error, val_error))
        return history

    def backpropagate(self, output, expected, debug):
        error = []
        for o, ex in zip(output, expected):
            if debug:
                print("Outputs: {} Expected: {}".format(round(o, 10), ex))
            error.append(o - ex)

        output_layer = self.layers[-1]
        if debug:
            print("ERROR ", error)
        gradients = output_layer.calculate_output_gradients(error)
        if debug:
            print("OUTPUT GRADIENTS ", gradients)

        for layer in reversed(self.layers[:-1]):
            layer.update_weights(gradients)
            gradients = layer.calculate_gradients(gradients)
            if debug:
                print("HIDDEN GRADIENTS ", gradients)

    def calculate_error(self, inputs, outputs):
        err_acum = 0
        for inp, out in zip(inputs, outputs):
            res = self.feedforward(inp)
            err_acum += sum([(r-o) ** 2 for r, o in zip(res, out)])

        return err_acum / len(inputs)

    def save(self, name, extremes):
        jdict = {}
        jdict["name"] = self.name
        jdict["layers"] = []
        jdict["extremes"] = []

        # if there are no layers save empty network
        if len(self.layers) < 1:
            with open(name, "w") as f:
                f.write(json.dumps(jdict))
            return

        if self.layers[0].has_bias:
            jdict["inputs"] = len(self.layers[0].neurons) - 1
        else:
            jdict["inputs"] = len(self.layers[0].neurons)

        for v in extremes:
            v = v.rstrip("\n")
            jdict["extremes"].append(v)

        for layer in self.layers:
            layer_dict = {}
            layer_dict["name"] = layer.name
            layer_dict["bias"] = layer.has_bias
            layer_dict["neurons"] = []
            for n in layer.neurons:
                neuron_dict = {}
                neuron_dict["connections"] = []
                for w in n.connections:
                    neuron_dict["connections"].append(n.connections[w].value)
                layer_dict["neurons"].append(neuron_dict.copy())
            jdict["layers"].append(layer_dict)

        with open(name, "w") as f:
            f.write(json.dumps(jdict))

    def load(self, name):
        # Remove any existing layers to start network from scratch
        self.layers = []

        with open(name, "r") as f:
            j = f.read()
            jdict = json.loads(j)

        num_inputs = int(jdict["inputs"])

        # For each input we need two values, minimum and maximum
        self.input_extremes = [float(x)
                               for x in jdict["extremes"][:num_inputs*2]]
        self.output_extremes = [float(x)
                                for x in jdict["extremes"][num_inputs*2:]]

        prev_conns_m = []

        for l in jdict["layers"]:
            input = len(self.layers) < 1
            if l["bias"]:
                new_layer = layer.Layer(len(l["neurons"])-1, l["name"], input)
            else:
                new_layer = layer.Layer(len(l["neurons"]), l["name"], input)
            if not input:
                prev_layer = self.layers[-1]
                prev_layer.connect_layer(new_layer)
                for n, conns in zip(prev_layer.neurons, prev_conns_m):
                    for i, nl_n in enumerate(new_layer.neurons):
                        n.connections[nl_n].value = conns[i]

            self.layers.append(new_layer)
            prev_conns_m = []
            for n in l["neurons"]:
                prev_conns_m.append(n["connections"])

        print("Model loaded")

    def normalize_input(self, input_row):
        normalized = []
        input_row = [float(x) for x in input_row.split(',')]
        for i, inp in enumerate(input_row):
            mini = self.input_extremes[i*2]
            maxi = self.input_extremes[i*2+1]
            normalized.append((inp - mini) / (maxi - mini))

        return normalized

    def denormalize_output(self, output_row):
        denormalized = []
        output_row = [float(x) for x in output_row]
        for i, inp in enumerate(output_row):
            mini = self.output_extremes[i*2]
            maxi = self.output_extremes[i*2+1]
            denormalized.append(inp * (maxi - mini) + mini)

        return denormalized
