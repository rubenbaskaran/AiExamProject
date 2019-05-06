import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt


class NeuralNetwork(object):

    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.x_input = None
        self.y_input = None
        self.hidden_layer_output = None
        self.error_x = []
        self.error_y = []
        self.global_error = 0
        self.counter = 0
        self.learning_rate = 0.5
        self.epochs = 1000
        self.input_size = 1
        self.hidden_size = 5
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
        self.x_input = np.array(data_from_csv["input"])  # ~~~~ Maybe need scaling? ~~~~
        self.y_input = np.array(data_from_csv["output"])  # ~~~~ Maybe need scaling? ~~~~

        # Weights
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
                predicted_output = self.forward_propagation(self.x_input[index])
                print("Predicted output: " + str(predicted_output))
                expected_output = self.sigmoid(self.y_input[index])
                print("Actual output: " + str(expected_output))
                error = (expected_output - predicted_output) ** 2  # ~~~~ Maybe modify this line ~~~~
                self.global_error += error
                self.back_propagation(input_value, expected_output, predicted_output)
            self.error_x.append(i)
            self.error_y.append(self.global_error/self.x_input.size)
            self.global_error = 0
            self.counter += 1
            print("Counter: " + str(self.counter))

    # forward-propagate the input in order to calculate an output
    def forward_propagation(self, input_value):
        z = np.dot(input_value, self.w1)  # dot product of input_value and first set of weights (1x3)
        print("Input · w1 = z -> " + str(z))
        z2 = self.sigmoid(z)  # insert dot product z into activation function
        self.hidden_layer_output = z2
        print("Run z through sigmoid = z2 -> " + str(z2))
        z3 = np.dot(z2, self.w2)  # dot product of hidden layer (z2) and second set of 3x1 weights
        print("z2 · w2 = z3 -> " + str(z3))
        prediction = self.sigmoid(z3)  # final activation function
        print("Run z3 through sigmoid = prediction -> " + str(prediction))
        return prediction[0][0]

    # back-propagate the error in order to train the network
    def back_propagation(self, input_value, expected_output, predicted_output):
        # error at output layer
        output_error = expected_output - predicted_output  # ~~~~ Maybe modify this line ~~~~

        # figure out how much to change weights between hidden layer and output layer
        output_delta = (output_error * self.sigmoid_prime(predicted_output))

        # error at hidden layer
        # Think of the error traveling back along the weights of the output layer to the neurons in the hidden layer
        hidden_layer_error = np.dot(output_delta, self.w2.T)

        # figure out how much to change weights between input layer and hidden layer
        hidden_layer_delta = hidden_layer_error * self.sigmoid_prime(self.hidden_layer_output)

        # Update weights
        self.w1 += np.dot(input_value.T, hidden_layer_delta) * self.learning_rate
        self.w2 += np.dot(self.hidden_layer_output.T, output_delta) * self.learning_rate

    def test_network(self):
        print("Starting test of network")
        print("W1: " + str(self.w1))
        print("W2: " + str(self.w2))
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


start = dt.datetime.now()
nn = NeuralNetwork()
nn.create_network()
nn.start_training()
print("\nStarted at: " + str(start) + "\n" + "Ended at: " + str(dt.datetime.now()))
nn.plot_error()
nn.test_network()
