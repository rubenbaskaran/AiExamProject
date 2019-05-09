import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D


class NeuralNetwork(object):

    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.x_input = None
        self.y_input = None
        self.L2_output = None
        self.error_x = []
        self.error_y = []
        self.global_error = 0
        self.counter = 0
        self.learning_rate = 0.5
        self.epochs = 10000
        self.input_size = 1
        self.hidden_size = 20
        self.output_size = 1

    def create_dataset(self):
        input = -1.0
        x_y_values = []
        z_values = []
        file = open("FunctionTwoDataset.csv", "w")
        file.write("input,output\n")
        StringBuilder = ""

        while input <= 1:
            x_y_values.append(input)
            output = np.exp(-(input ** 2 + input ** 2) / 0.1)
            z_values.append(output)
            StringBuilder += str(input) + "," + str(output) + "\n"
            input += 0.05

        file.write(StringBuilder)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(-1, 1, 0.05)
        Y = np.arange(-1, 1, 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = np.exp(-(X ** 2 + Y ** 2) / 0.1)

        ax.plot_surface(X, Y, Z)
        ax.set_zlim(0, 1)
        plt.show()

    def create_network(self):
        # Import data
        data_from_csv = pd.read_csv('FunctionTwoDataset.csv')
        self.x_input = np.array(data_from_csv["input"])  # ~~~~ Maybe need scaling? ~~~~
        self.y_input = np.array(data_from_csv["output"])  # ~~~~ Maybe need scaling? ~~~~

        # Weights
        self.w1 = np.random.randn(self.input_size, self.hidden_size)  # (1x3) weight matrix from input to hidden layer
        self.w2 = np.random.randn(self.hidden_size, self.output_size)  # (3x1) weight matrix from hidden to output layer

    def start_training(self):
        for i in range(self.epochs):
            for index in range(0, self.x_input.size):
                predicted_output = self.forward_propagation(self.x_input[index])
                expected_output = self.sigmoid(self.y_input[index])
                error = (expected_output - predicted_output) ** 2  # ~~~~ Maybe modify this line ~~~~
                self.global_error += error
                self.back_propagation(self.x_input[index], expected_output, predicted_output)
            self.error_x.append(i)
            self.error_y.append(self.global_error / self.x_input.size)
            self.global_error = 0
            self.counter += 1
            print("Epoch: " + str(self.counter) + "/" + str(self.epochs))

    # forward-propagate the input in order to calculate an output
    def forward_propagation(self, input_value):
        self.L2_output = self.sigmoid(np.dot(input_value, self.w1))
        prediction = self.sigmoid(np.dot(self.L2_output, self.w2))
        return prediction[0][0]

    # back-propagate the error in order to train the network
    # Using partial derivative and chain-rule
    def back_propagation(self, input_value, expected_output, predicted_output):
        # Figure out how much W2 contributed to output error
        # And how much to change W2
        L3_error = expected_output - predicted_output
        w2_delta = (L3_error * self.sigmoid_prime(predicted_output))

        # Figure out how much W1 contributed to output error
        # And how much to change W1
        L2_error = np.dot(w2_delta, self.w2.T)
        w1_delta = L2_error * self.sigmoid_prime(self.L2_output)

        # Update weights
        self.w1 += np.dot(input_value.T, w1_delta) * self.learning_rate
        self.w2 += np.dot(self.L2_output.T, w2_delta) * self.learning_rate

    def test_network(self):
        x_values = []
        y_values_predicted = []
        y_values_actual = []

        for index in range(0, self.x_input.size):
            x_values.append(self.x_input[index])
            y_values_predicted.append(self.forward_propagation(self.x_input[index]))
            y_values_actual.append(self.sigmoid(self.y_input[index]))

        figure = plt.figure()
        axes = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(x_values, y_values_predicted)
        axes.plot(x_values, y_values_actual)
        axes.set_title("Actual model vs. trained model (3 Layers)")
        plt.show()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Given an output value from a neuron, we want to calculate it’s slope
    def sigmoid_prime(self, x):
        return x * (1 - x)

    def plot_error(self):
        plt.plot(self.error_x, self.error_y)
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared error")
        plt.title("Mean squared error for each epoch (3 Layers)")
        plt.show()


# print("Started at: " + str(dt.datetime.now()))
nn = NeuralNetwork()
# nn.create_network()
# nn.start_training()
# print("Ended at: " + str(dt.datetime.now()))
# nn.plot_error()
# nn.test_network()
nn.create_dataset()