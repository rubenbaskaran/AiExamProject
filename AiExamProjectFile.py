import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class NeuralNetwork(object):

    def create_dataset(self):
        input = -1.0
        x_list = []
        y_list = []
        f = open("dataset.csv", "w")
        f.write("input,output\n")
        StringBuilder = ""

        while input <= 1:
            x_list.append(input)
            output = round(math.sin(2 * math.pi * input) + math.sin(5 * math.pi * input), 3)
            y_list.append(output)
            StringBuilder = StringBuilder + str(input) + "," + str(output) + "\n"
            input = round(input + 0.002, 3)

        f.write(StringBuilder)
        plt.plot(x_list, y_list)
        plt.axis([-1, 1, -2, 2])
        plt.show()

    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.x_input = None
        self.y_input = None
        self.output_at_hidden_layer = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def create_network(self):
        # Import data
        data_from_csv = pd.read_csv('dataset.csv')
        self.x_input = np.array(data_from_csv["input"])
        self.y_input = np.array(data_from_csv["output"])

        # Network structure
        input_size = 1
        output_size = 1
        hidden_size = 3

        # Weights
        np.random.seed(1)
        self.w1 = np.random.randn(input_size, hidden_size)  # (1x3) weight matrix from input to hidden layer
        self.w2 = np.random.randn(hidden_size, output_size)  # (3x1) weight matrix from hidden to output layer

        self.start_training()

    def start_training(self):
        # for index in range(0, x_input.size):
        for index in range(0, 2):
            print("W1: " + str(self.w1))
            print("W2: " + str(self.w2))
            input_value = self.x_input[index]
            print("Input: " + str(input_value))
            output_value = self.forward_propagation(self.x_input[index])
            print("Predicted output: " + str(output_value))
            actual_output = self.sigmoid(self.y_input[index])
            print("Actual output: " + str(actual_output))
            #self.back_propagation(input_value, actual_output, output_value)

    def forward_propagation(self, input_value):
        z = np.dot(input_value, self.w1)  # dot product of input_value and first set of weights (1x3)
        print("Input · w1 = z -> " + str(z))
        z2 = self.sigmoid(z)  # insert dot product z into activation function
        self.output_at_hidden_layer = z2
        print("Run z through sigmoid = z2 -> " + str(z2))
        z3 = np.dot(z2, self.w2)  # dot product of hidden layer (z2) and second set of 3x1 weights
        print("z2 · w2 = z3 -> " + str(z3))
        prediction = self.sigmoid(z3)  # final activation function
        print("Run z3 through sigmoid = prediction -> " + str(prediction))
        return prediction

    def back_propagation(self, input_value, actual_output, output_at_output_layer):
        layer2_error = actual_output - output_at_output_layer
        layer2_delta = layer2_error * self.sigmoid_prime(output_at_output_layer)
        layer1_error = np.dot(layer2_delta, self.w2.T)
        layer1_delta = layer1_error * self.sigmoid_prime(self.output_at_hidden_layer)

        # Update weights
        self.w2 = np.dot(self.output_at_hidden_layer.T, layer2_delta)
        self.w1 = np.dot(input_value.T, layer1_delta)

        print("Error: " + str(layer2_error))


nn = NeuralNetwork()
nn.create_network()
