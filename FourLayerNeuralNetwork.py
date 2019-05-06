import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt


class NeuralNetwork(object):

    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.x_input = None
        self.y_input = None
        self.L2_output = None
        self.L3_output = None
        self.error_x = []
        self.error_y = []
        self.global_error = 0
        self.counter = 0
        self.learning_rate = 0.5
        self.epochs = 500
        self.input_size = 1
        self.first_hidden_size = 5
        self.second_hidden_size = 5
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
        self.w1 = np.random.randn(self.input_size, self.first_hidden_size)          # (1x5) weight matrix from input to hidden layer
        self.w2 = np.random.randn(self.first_hidden_size, self.second_hidden_size)  # (5x5) weight matrix from hidden to output layer
        self.w3 = np.random.randn(self.second_hidden_size, self.output_size)        # (5x1) weight matrix from hidden to output layer

    def start_training(self):
        for i in range(self.epochs):
            for index in range(0, self.x_input.size):
                # for index in range(0, 2):
                print("W1: " + str(self.w1))
                print("W2: " + str(self.w2))
                print("W3: " + str(self.w3))
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
        self.L2_output = self.sigmoid(np.dot(input_value, self.w1))
        print("sigmoid(input_value · w1) = L2_output -> " + str(self.L2_output))
        self.L3_output = self.sigmoid(np.dot(self.L2_output, self.w2))
        print("sigmoid(L2_output · w2 = L3_output -> " + str(self.L3_output))
        prediction = self.sigmoid(np.dot(self.L3_output, self.w3))
        print("sigmoid(L3_output · w3 = prediction -> " + str(prediction))
        return prediction[0][0]

    # back-propagate the error in order to train the network
    def back_propagation(self, input_value, expected_output, predicted_output):
        # Figure out how much to change W3
        L4_error = expected_output - predicted_output
        w3_delta = L4_error * self.sigmoid_prime(predicted_output)

        # Figure out how much to change W2
        L3_error = np.dot(w3_delta, self.w3.T)
        w2_delta = L3_error * self.sigmoid_prime(self.L3_output)

        # Figure out how much to change W1
        L2_error = np.dot(w2_delta, self.w2.T)
        w1_delta = L2_error * self.sigmoid_prime(self.L2_output)

        # Update weights
        self.w1 += np.dot(input_value.T, w1_delta) * self.learning_rate
        self.w2 += np.dot(self.L2_output.T, w2_delta) * self.learning_rate
        self.w3 += np.dot(self.L3_output.T, w3_delta) * self.learning_rate

    def test_network(self):
        print("Starting test of network")
        print("W1: " + str(self.w1))
        print("W2: " + str(self.w2))
        print("W3: " + str(self.w3))
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
        axes.set_title("Actual model vs. trained model")
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
        plt.title("Mean squared error for each epoch")
        plt.show()


start = dt.datetime.now()
nn = NeuralNetwork()
nn.create_network()
nn.start_training()
nn.plot_error()
nn.test_network()
print("\nStarted at: " + str(start) + "\n" + "Ended at: " + str(dt.datetime.now()))
