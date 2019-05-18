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
        self.train_data = None
        self.test_data = None
        self.x_y_input = None
        self.z_output = None
        self.L2_output = None
        self.error_x_axis = []
        self.error_y_axis = []
        self.global_error = 0
        self.counter = 1
        self.learning_rate = 0.5
        self.epochs = epochs
        self.input_size = 2
        self.hidden_size = hidden_size
        self.output_size = 1
        self.timestamp_start = dt.datetime.now()
        self.execution_time = None
        self.mse = None

        # self.create_dataset()
        self.create_network()
        self.start_training()
        self.add_to_statistics(csv_writer)

    def add_to_statistics(self, csv_writer):
        csv_writer.write("\n" + str(self.hidden_size) + "," + str(self.epochs) + "," + str(self.execution_time) + "," + str(self.mse))

    def create_dataset(self):
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "FunctionTwoDataset.csv")
        file = open(path, "w")
        file.write("x_input,y_input,z_output\n")
        string_builder = ""
        z_values = []
        x_values = y_values = np.round(np.arange(-1, 1.05, 0.05), 3)

        for x_value in x_values:
            for y_value in y_values:
                z_value = np.round(np.exp(-(x_value ** 2 + y_value ** 2) / 0.1), 3)
                z_values.append(z_value)
                string_builder += str(x_value) + "," + str(y_value) + "," + str(z_value) + "\n"
        file.write(string_builder)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_values, y_values = np.meshgrid(x_values, y_values)
        ax.plot_surface(x_values, y_values, np.array(z_values).reshape(41, 41))
        ax.set_title("Actual model")
        ax.set_zlim(0, 1)
        plt.show()

    def create_network(self):
        # Import data
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "FunctionTwoDataset.csv")
        data_from_csv = pd.read_csv(path)
        x_values = np.round(np.array(data_from_csv["x_input"]), 3)
        y_values = np.round(np.array(data_from_csv["y_input"]), 3)
        self.z_output = np.round(np.array(data_from_csv["z_output"]), 3)

        x_y_input_builder = []
        for index in range(0, x_values.size):
            x_y_input_builder.append([x_values[index], y_values[index], self.z_output[index]])
        self.x_y_input = np.array(x_y_input_builder)
        np.random.seed(1)
        np.random.shuffle(self.x_y_input)
        self.train_data = self.x_y_input[:1177]
        self.test_data = self.x_y_input[1177:]

        # Weights
        self.w1 = np.random.randn(self.input_size, self.hidden_size)    # (2x20) between input and hidden
        self.w2 = np.random.randn(self.hidden_size, self.output_size)   # (20x1) between hidden and output

    def start_training(self):
        for i in range(self.epochs):
            for x_y_z_data in self.train_data:
                x_y_input = np.array([x_y_z_data[0], x_y_z_data[1]])
                predicted_output = self.forward_propagation(x_y_input)
                expected_output = x_y_z_data[2]
                error = (expected_output - predicted_output) ** 2
                self.global_error += error
                self.back_propagation(x_y_input, expected_output, predicted_output)
            self.error_x_axis.append(i)
            self.error_y_axis.append(self.global_error / len(self.train_data))
            self.global_error = 0
            print("Epoch: " + str(self.counter) + "/" + str(self.epochs) + " (FunctionTwoSingleHidden)")
            self.counter += 1
        self.mse = np.round(self.error_y_axis.__getitem__(len(self.error_y_axis) - 1)[0], 10)
        self.execution_time = str(dt.datetime.now() - self.timestamp_start).split('.')[0]

    # forward-propagate the input in order to calculate an output
    def forward_propagation(self, input_value):
        self.L2_output = self.sigmoid(np.dot(input_value, self.w1))
        prediction = np.dot(self.L2_output, self.w2)
        return prediction

    # back-propagate the error in order to train the network
    # Using partial derivative and chain-rule
    def back_propagation(self, input_value, expected_output, predicted_output):
        # Figure out how much W2 contributed to output error
        # And how much to change W2
        L3_error = expected_output - predicted_output
        w2_delta = L3_error * self.sigmoid_prime(self.sigmoid(predicted_output))

        # Figure out how much W1 contributed to output error
        # And how much to change W1
        L2_error = np.dot(w2_delta, self.w2.T)
        w1_delta = L2_error * self.sigmoid_prime(self.L2_output)

        # Update weights
        self.w1 += np.dot(np.array([input_value]).T, np.array([w1_delta])) * self.learning_rate
        self.w2 += np.dot(np.array([self.L2_output]).T, np.array([w2_delta])) * self.learning_rate

    def plot_error(self):
        plt.plot(self.error_x_axis, self.error_y_axis)
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared error")
        plt.title("Mean squared error for each epoch (1 hidden layer)" + "\nMSE: " + str(self.error_y_axis.__getitem__(len(self.error_y_axis) - 1)[0]))
        plt.show()

    def test_network(self):
        x_values = []
        y_values = []
        z_values_predicted = []
        z_values_actual = []

        for x_y_z_data in self.test_data:
            x_values.append(x_y_z_data[0])
            y_values.append(x_y_z_data[1])
            x_y_input = np.array([x_y_z_data[0], x_y_z_data[1]])
            z_values_predicted.append(self.forward_propagation(x_y_input)[0])
            z_values_actual.append(x_y_z_data[2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_values, y_values, z_values_actual, color="royalblue")
        ax.scatter(x_values, y_values, z_values_predicted, color="orange")

        col1_patch = mpatches.Patch(color="royalblue", label='Actual')
        col2_patch = mpatches.Patch(color="orange", label='Predicted')
        plt.legend(handles=[col1_patch, col2_patch], loc=(0.0, 0.7))

        ax.set_title("Actual model vs. trained model (1 hidden layer)" + "\nMSE: " + str(self.error_y_axis.__getitem__(len(self.error_y_axis) - 1)[0]))
        plt.xlabel("x-values")
        plt.ylabel("y-values")
        ax.set_zlim(0, 1)
        plt.show()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Given an output value from a neuron, we want to calculate it’s slope
    def sigmoid_prime(self, x):
        return x * (1 - x)
