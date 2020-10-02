import numpy as np


def sigmoid(x):
    # sigmoid function for neural desion making
    return 1./(1.+np.exp(-x))


class Neuron():
    def __init__(self):
        self.value = 0
        self.activated = False
        self.input_values = []
        self.inputs = 0
        self.other_neurons = []
        self.bias = 0

    def add_neuron(self, other_neuron, weight, bias):
        other_neuron.inputs += 1
        other_neuron.bias = bias
        self.other_neurons.append((other_neuron, weight, bias))

    def activate(self):
        self.value = sigmoid(sum(self.input_values) + self.bias)
        self.activated = True

    def calculate_connections(self):
        for (neuron, weight, bias) in self.other_neurons:
            neuron.input_values.append(self.value * weight)


class Controller:
    def __init__(self):
        self.n_outputs = 5

    def init_network(self, genes):
        neurons = []
        # find max innovation number to determine the total amount of neurons
        max_innovation_n = 0

        for gene in genes:

            if gene.a > max_innovation_n:
                max_innovation_n = gene.a
            if gene.b > max_innovation_n:
                max_innovation_n = gene.b
        # initialise neurons
        for i in range(0, max_innovation_n + 1):
            neurons.append(Neuron())
        # add connections
        for gene in genes:
            neurons[gene.a].add_neuron(neurons[gene.b], gene.weight, gene.bias)

        return neurons

    def control(self, inputs, genes):
        # normalise input, taken from demo_controller
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        neurons = self.init_network(genes)
        # activate input neurons
        for i in range(0, len(inputs)):
            neurons[i].value = inputs[i]
            neurons[i].activated = True
            neurons[i].calculate_connections()

        # keep running until all output neurons are activated:
        continue_bool = True
        while continue_bool:
            for i in range(0, len(neurons)):
                if not neurons[i].activated:
                    if len(neurons[i].input_values) == neurons[i].inputs:
                        neurons[i].activate()
                        neurons[i].calculate_connections()

            # break if all output values are activated
            continue_bool = False
            for i in range(len(inputs),len(inputs) + self.n_outputs):
                if not neurons[i].activated:
                    continue_bool = True


        # create output array
        action_array = []
        for output in range(len(inputs),len(inputs) + self.n_outputs):
            action_array.append(int(neurons[output].value > 0.5))

        return action_array
