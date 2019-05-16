import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.patches as mpatches
import os.path
from mpl_toolkits.mplot3d import Axes3D


class NeuralNetwork(object):

    def __init__(self, epochs, hidden_size, csv_writer):
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.x_values = np.round(np.arange(-1, 1.05, 0.05), 3)
        self.y_values = np.round(np.arange(-1, 1.05, 0.05), 3)
        self.x_y_input = None
        self.z_output = None
        self.L2_output = None
        self.L3_output = None
        self.error_x_y = []
        self.error_z = []
        self.global_error = 0
        self.counter = 1
        self.learning_rate = 0.5
        self.epochs = epochs
        self.input_size = 2
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
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "FunctionTwoDataset.csv")
        file = open(path, "w")
        file.write("x_input,y_input,z_output\n")
        string_builder = ""
        z_values = []

        for x_value in self.x_values:
            for y_value in self.y_values:
                z_value = np.exp(-(x_value ** 2 + y_value ** 2) / 0.1)
                z_values.append(z_value)
                string_builder += str(x_value) + "," + str(y_value) + "," + str(z_value) + "\n"
        file.write(string_builder)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_values, y_values = np.meshgrid(self.x_values, self.y_values)
        ax.plot_surface(x_values, y_values, np.array(z_values).reshape(41, 41))
        ax.set_title("Actual model")
        ax.set_zlim(0, 1)
        plt.show()

    def create_network(self):
        # Import data
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "FunctionTwoDataset.csv")
        data_from_csv = pd.read_csv(path)
        x_values = np.array(data_from_csv["x_input"])
        y_values = np.array(data_from_csv["y_input"])
        self.z_output = np.array(data_from_csv["z_output"])

        x_y_input_builder = []
        for index in range(0, x_values.size):
            x_y_input_builder.append([x_values[index], y_values[index]])
        self.x_y_input = np.array(x_y_input_builder)

        # Weights
        self.w1 = np.random.randn(self.input_size, self.first_hidden_size)          # (2x100) between input and first hidden
        self.w2 = np.random.randn(self.first_hidden_size, self.second_hidden_size)  # (100x100) between first hidden and second hidden
        self.w3 = np.random.randn(self.second_hidden_size, self.output_size)        # (100x1) between second hidden and output

    def start_training(self):
        for i in range(self.epochs):
            for index in range(0, len(self.x_y_input)):
                predicted_output = self.forward_propagation(self.x_y_input[index])
                expected_output = self.sigmoid(self.z_output[index])
                error = (expected_output - predicted_output) ** 2
                self.global_error += error
                self.back_propagation(self.x_y_input[index], expected_output, predicted_output)
            self.error_x_y.append(i)
            self.error_z.append(self.global_error / self.x_y_input.size)
            self.global_error = 0
            print("Epoch: " + str(self.counter) + "/" + str(self.epochs) + " (FunctionTwoMultiHidden)")
            self.counter += 1
        self.mse = self.error_z.__getitem__(len(self.error_z) - 1)[0]
        self.execution_time = dt.datetime.now() - self.timestamp_start

    # forward-propagate the input in order to calculate an output
    def forward_propagation(self, input_value):
        self.L2_output = self.sigmoid(np.dot(input_value, self.w1))
        self.L3_output = self.sigmoid(np.dot(self.L2_output, self.w2))
        prediction = self.sigmoid(np.dot(self.L3_output, self.w3))
        return prediction

    # back-propagate the error in order to train the network
    # Using partial derivative and chain-rule
    def back_propagation(self, input_value, expected_output, predicted_output):
        # Figure out how much W3 contributed to output error
        # And how much to change W3
        L4_error = expected_output - predicted_output
        w3_delta = L4_error * self.sigmoid_prime(predicted_output)

        # Figure out how much W2 contributed to output error
        # And how much to change W2
        L3_error = np.dot(w3_delta, self.w3.T)
        w2_delta = L3_error * self.sigmoid_prime(self.L3_output)

        # Figure out how much W1 contributed to output error
        # And how much to change W1
        L2_error = np.dot(w2_delta, self.w2.T)
        w1_delta = L2_error * self.sigmoid_prime(self.L2_output)

        # Update weights
        self.w1 += np.dot(np.array([input_value]).T, np.array([w1_delta])) * self.learning_rate
        self.w2 += np.dot(np.array([self.L2_output]).T, np.array([w2_delta])) * self.learning_rate
        self.w3 += np.dot(np.array([self.L3_output]).T, np.array([w3_delta])) * self.learning_rate

    def plot_error(self):
        plt.plot(self.error_x_y, self.error_z)
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared error")
        plt.title("Mean squared error for each epoch (2 hidden layers)" + "\nMSE: " + str(self.error_z.__getitem__(len(self.error_z) - 1)[0]))
        plt.show()

    def test_network(self):
        z_values_predicted = []
        z_values_actual = []

        for index in range(0, len(self.x_y_input)):
            z_values_predicted.append(self.forward_propagation(self.x_y_input[index])[0])
            z_values_actual.append(self.z_output[index])

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_values, y_values = np.meshgrid(self.x_values, self.y_values)
        ax.plot_surface(x_values, y_values, np.array(z_values_actual).reshape(41, 41), color="royalblue")
        ax.plot_surface(x_values, y_values, np.array(z_values_predicted).reshape(41, 41), color="orange")

        col1_patch = mpatches.Patch(color="royalblue", label='Actual')
        col2_patch = mpatches.Patch(color="orange", label='Predicted')
        plt.legend(handles=[col1_patch, col2_patch], loc=(0.0, 0.7))

        ax.set_title("Actual model vs. trained model (2 hidden layers)" + "\nMSE: " + str(self.error_z.__getitem__(len(self.error_z) - 1)[0]))
        plt.xlabel("x-values")
        plt.ylabel("y-values")
        ax.set_zlim(0, 1)
        plt.show()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Given an output value from a neuron, we want to calculate it’s slope
    def sigmoid_prime(self, x):
        return x * (1 - x)
