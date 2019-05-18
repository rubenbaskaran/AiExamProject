import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import os.path


class NeuralNetwork(object):

    def __init__(self, epochs, hidden_size, csv_writer):
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.train_data = None
        self.test_data = None
        self.x_input = None
        self.y_output = None
        self.L2_output = None
        self.L3_output = None
        self.error_x_axis = []
        self.error_y_axis = []
        self.global_error = 0
        self.counter = 1
        self.learning_rate = 0.5
        self.epochs = epochs
        self.input_size = 1
        self.first_hidden_size = hidden_size
        self.second_hidden_size = hidden_size
        self.output_size = 1
        self.timestamp_start = dt.datetime.now()
        self.execution_time = None
        self.mse = None

        self.create_network()
        self.start_training()
        self.add_to_statistics(csv_writer)

    def add_to_statistics(self, csv_writer):
        csv_writer.write("\n" + str(self.first_hidden_size) + "," + str(self.epochs) + "," + str(self.execution_time) + "," + str(self.mse))

    def create_dataset(self):
        input_values = np.round(np.arange(-1, 1.002, 0.002), 3)
        x_list = []
        y_list = []
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "FunctionOneDataset.csv")
        file = open(path, "w")
        file.write("input,output\n")
        string_builder = ""

        for input_value in input_values:
            x_list.append(input_value)
            output = np.round(math.sin(2 * math.pi * input_value) + math.sin(5 * math.pi * input_value), 3)
            y_list.append(output)
            string_builder += str(input_value) + "," + str(output) + "\n"

        file.write(string_builder)
        plt.plot(x_list, y_list)
        plt.axis([-1, 1, -2, 2])
        plt.show()

    def create_network(self):
        # Import data
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "FunctionOneDataset.csv")
        data_from_csv = pd.read_csv(path)
        self.x_input = np.round(np.array(data_from_csv["input"]), 3)
        self.y_output = np.round(np.array(data_from_csv["output"]), 3)

        x_y_dataset_builder = []
        for index in range(0, self.x_input.size):
            x_y_dataset_builder.append([self.x_input[index], self.y_output[index]])
        np.random.seed(1)
        np.random.shuffle(x_y_dataset_builder)
        self.train_data = x_y_dataset_builder[:700]
        self.test_data = x_y_dataset_builder[700:]

        # Weights
        self.w1 = np.random.randn(self.input_size, self.first_hidden_size)          # (1x5) weight matrix from input to hidden layer
        self.w2 = np.random.randn(self.first_hidden_size, self.second_hidden_size)  # (5x5) weight matrix from hidden to output layer
        self.w3 = np.random.randn(self.second_hidden_size, self.output_size)        # (5x1) weight matrix from hidden to output layer

    def start_training(self):
        for i in range(self.epochs):
            for x_y_data in self.train_data:
                predicted_output = self.forward_propagation(x_y_data[0])
                expected_output = x_y_data[1]
                error = (expected_output - predicted_output) ** 2
                self.global_error += error
                self.back_propagation(x_y_data[0], expected_output, predicted_output)
            self.error_x_axis.append(i)
            self.error_y_axis.append(self.global_error / len(self.train_data))
            self.global_error = 0
            print("Epoch: " + str(self.counter) + "/" + str(self.epochs) + " (FunctionOneMultiHidden)")
            self.counter += 1
        self.mse = np.round(self.error_y_axis.__getitem__(len(self.error_y_axis) - 1), 10)
        self.execution_time = str(dt.datetime.now() - self.timestamp_start).split('.')[0]

    # forward-propagate the input in order to calculate an output
    def forward_propagation(self, input_value):
        self.L2_output = self.sigmoid(np.dot(input_value, self.w1))
        self.L3_output = self.sigmoid(np.dot(self.L2_output, self.w2))
        prediction = np.dot(self.L3_output, self.w3)
        return prediction[0][0]

    # back-propagate the error in order to train the network
    # Using partial derivative and chain-rule
    def back_propagation(self, input_value, expected_output, predicted_output):
        # Figure out how much W3 contributed to output error
        # And how much to change W3
        L4_error = expected_output - predicted_output
        w3_delta = L4_error * self.sigmoid_prime(self.sigmoid(predicted_output))

        # Figure out how much W2 contributed to output error
        # And how much to change W2
        L3_error = np.dot(w3_delta, self.w3.T)
        w2_delta = L3_error * self.sigmoid_prime(self.L3_output)

        # Figure out how much W1 contributed to output error
        # And how much to change W1
        L2_error = np.dot(w2_delta, self.w2.T)
        w1_delta = L2_error * self.sigmoid_prime(self.L2_output)

        # Update weights
        self.w1 += np.dot(input_value.T, w1_delta) * self.learning_rate
        self.w2 += np.dot(self.L2_output.T, w2_delta) * self.learning_rate
        self.w3 += np.dot(self.L3_output.T, w3_delta) * self.learning_rate

    def plot_error(self):
        plt.plot(self.error_x_axis, self.error_y_axis)
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared error")
        plt.title("Mean squared error for each epoch (2 hidden layers)" + "\nMSE: " + str(self.error_y_axis.__getitem__(len(self.error_y_axis) - 1)))
        plt.show()

    def test_network(self):
        x_values = []
        y_values_predicted = []
        y_values_actual = []

        for x_y_data in self.test_data:
            x_values.append(x_y_data[0])
            y_values_predicted.append(self.forward_propagation(x_y_data[0]))
            y_values_actual.append(x_y_data[1])

        figure = plt.figure()
        axes = figure.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.scatter(x_values, y_values_actual, label="Actual", color="royalblue", s=10)
        axes.scatter(x_values, y_values_predicted, label="Predicted", color="orange", s=10)
        axes.legend(loc="upper right")
        plt.xlabel("x-values")
        plt.ylabel("y-values")
        axes.set_title("Actual model vs. trained model (2 hidden layers)" + "\nMSE: " + str(self.error_y_axis.__getitem__(len(self.error_y_axis) - 1)))
        plt.show()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Given an output value from a neuron, we want to calculate it’s slope
    def sigmoid_prime(self, x):
        return x * (1 - x)
