import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class NeuralNetwork(object):

    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.x_input = None
        self.y_input = None
        self.output_at_hidden_layer = None
        self.error_x = []
        self.error_y = []
        self.counter = 0
        self.learning_rate = 0.1
        self.epochs = 10
        self.input_size = 1
        self.hidden_size = 3
        self.output_size = 1

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

    def create_network(self):
        # Import data
        data_from_csv = pd.read_csv('dataset.csv')
        self.x_input = np.array(data_from_csv["input"])
        self.y_input = np.array(data_from_csv["output"])

        # Weights
        np.random.seed(1)
        self.w1 = np.random.randn(self.input_size, self.hidden_size)  # (1x3) weight matrix from input to hidden layer
        self.w2 = np.random.randn(self.hidden_size, self.output_size)  # (3x1) weight matrix from hidden to output layer

    def start_training(self):
        for i in range(self.epochs):
            for index in range(0, self.x_input.size):
                # for index in range(0, 2):
                print("W1: " + str(self.w1))
                print("W2: " + str(self.w2))
                input_value = self.x_input[index]
                print("Input: " + str(input_value))
                output_value = self.forward_propagation(self.x_input[index])
                print("Predicted output: " + str(output_value))
                actual_output = self.sigmoid(self.y_input[index])
                print("Actual output: " + str(actual_output))
                self.error_x.append(self.counter)
                error = np.mean(np.square(actual_output - output_value))
                self.error_y.append(error)
                self.counter = self.counter + 1
                self.back_propagation(input_value, actual_output, output_value)
        print("Counter: " + str(self.counter))

    # forward-propagate the input in order to calculate an output
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

    # back-propagate the error in order to train the network
    def back_propagation(self, input_value, expected_output, predicted_output):
        # error at output layer
        output_error = expected_output - predicted_output

        # figure out how much to change weights between hidden layer and output layer
        output_delta = (output_error * self.sigmoid_prime(predicted_output))

        # error at hidden layer
        hidden_layer_error = np.dot(output_delta, self.w2.T)

        # figure out how much to change weights between input layer and hidden layer
        hidden_layer_delta = hidden_layer_error * self.sigmoid_prime(self.output_at_hidden_layer)

        # Update weights
        self.w1 += np.dot(input_value.T, hidden_layer_delta)
        self.w2 += np.dot(self.output_at_hidden_layer.T, output_delta)

    def test_network(self):
        print("Starting test of network")
        print("W1: " + str(self.w1))
        print("W2: " + str(self.w2))
        x_values = []
        y_values_predicted = []
        y_values_actual = []

        for index in range(0, self.x_input.size):
            x_values.append(self.x_input[index])
            y_values_predicted.append(self.forward_propagation(self.x_input[index])[0][0])
            y_values_actual.append(self.sigmoid(self.y_input[index]))

        figure = plt.figure()
        axes = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(x_values, y_values_predicted)
        axes.plot(x_values, y_values_actual)
        axes.set_title("Actual model vs. trained model")
        plt.show()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def plot_error(self):
        plt.plot(self.error_x, self.error_y)
        plt.xlabel("Training sample")
        plt.ylabel("Error")
        plt.title("Error for each training sample")
        plt.show()


nn = NeuralNetwork()
nn.create_network()
nn.start_training()
nn.plot_error()
nn.test_network()
