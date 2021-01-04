import layer
import neural_network as nn
import read_data as rd
import matplotlib.pyplot as plt

network = nn.NeuralNetwork("My Network")

network.add(layer.Layer(2, "Input"))
network.add(layer.Layer(5, "Hidden 1"))
network.add(layer.Layer(2, "Output"))

print(network)

inputs, outputs = rd.read_data("processed.csv", 2)
extremes = rd.read_extremes("processed.csv")
print("Predicting {} = {}".format(
    outputs[-2000], network.feedforward(inputs[-2000])))

x_train = inputs[:int(len(inputs) * 0.7)]
y_train = outputs[:int(len(inputs) * 0.7)]

x_val = inputs[int(len(inputs) * 0.7):]
y_val = outputs[int(len(inputs) * 0.7):]

history = network.train(x_train, y_train, x_val, y_val, 10, False)

plt.plot(history[0], history[1], "b-", label="Training Error")
plt.plot(history[0], history[2], "m-", label="Validation Error")
plt.title("Training Process")
plt.legend()
plt.xlabel = "Epoch"
plt.ylabel = "Error"

print(network)

print("Predicting {} = {}".format(
    outputs[-2000], network.feedforward(inputs[-2000])))

network.save("model.json", extremes)

plt.show()
