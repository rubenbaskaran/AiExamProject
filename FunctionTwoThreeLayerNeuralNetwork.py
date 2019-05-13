import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D


class NeuralNetwork(object):

    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.x_values = np.round(np.arange(-1, 1.05, 0.05), 3)
        self.y_values = np.round(np.arange(-1, 1.05, 0.05), 3)
        self.x_y_input = None
        self.z_output = None
        self.L2_output = None
        self.error_x_y = []
        self.error_z = []
        self.global_error = 0
        self.counter = 1
        self.learning_rate = 1
        self.epochs = 1000
        self.input_size = 2
        self.hidden_size = 100
        self.output_size = 1

    def create_dataset(self):
        file = open("FunctionTwoDataset.csv", "w")
        file.write("x_input,y_input,z_output\n")
        StringBuilder = ""
        z_values = []

        for x_value in self.x_values:
            for y_value in self.y_values:
                z_value = np.exp(-(x_value ** 2 + y_value ** 2) / 0.1)
                z_values.append(z_value)
                StringBuilder += str(x_value) + "," + str(y_value) + "," + str(z_value) + "\n"
        file.write(StringBuilder)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_values, y_values = np.meshgrid(self.x_values, self.y_values)
        ax.plot_surface(x_values, y_values, np.array(z_values).reshape(41, 41))
        ax.set_title("Actual model")
        ax.set_zlim(0, 1)
        plt.show()

    def create_network(self):
        # Import data
        data_from_csv = pd.read_csv('FunctionTwoDataset.csv')
        x_values = np.array(data_from_csv["x_input"])
        y_values = np.array(data_from_csv["y_input"])
        self.z_output = np.array(data_from_csv["z_output"])

        x_y_input_builder = []
        for index in range(0, x_values.size):
            x_y_input_builder.append([x_values[index], y_values[index]])
        self.x_y_input = np.array(x_y_input_builder)

        # Weights
        self.w1 = np.random.randn(self.input_size, self.hidden_size)    # (2x20) between input and hidden
        self.w2 = np.random.randn(self.hidden_size, self.output_size)   # (20x1) between hidden and output

    def start_training(self):
        for i in range(self.epochs):
            for index in range(0, len(self.x_y_input)):
                predicted_output = self.forward_propagation(self.x_y_input[index])
                expected_output = self.sigmoid(self.z_output[index])
                error = (expected_output - predicted_output) ** 2
                self.global_error += error
                self.back_propagation(self.x_y_input[index], expected_output, predicted_output)
            self.error_x_y.append(i)
            self.error_z.append(self.global_error / len(self.x_y_input))
            self.global_error = 0
            print("Epoch: " + str(self.counter) + "/" + str(self.epochs))
            self.counter += 1

    # forward-propagate the input in order to calculate an output
    def forward_propagation(self, input_value):
        self.L2_output = self.sigmoid(np.dot(input_value, self.w1))
        prediction = self.sigmoid(np.dot(self.L2_output, self.w2))
        return prediction

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
        self.w1 += np.dot(np.array([input_value]).T, np.array([w1_delta])) * self.learning_rate
        self.w2 += np.dot(np.array([self.L2_output]).T, np.array([w2_delta])) * self.learning_rate

    def test_network(self):
        z_values_predicted = []
        z_values_actual = []

        for index in range(0, len(self.x_y_input)):
            z_values_predicted.append(self.forward_propagation(self.x_y_input[index])[0])
            z_values_actual.append(self.z_output[index])

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_values, y_values = np.meshgrid(self.x_values, self.y_values)
        ax.plot_surface(x_values, y_values, np.array(z_values_actual).reshape(41, 41))
        ax.plot_surface(x_values, y_values, np.array(z_values_predicted).reshape(41, 41))
        ax.set_title("Actual model vs. trained model (3 Layers)")
        ax.set_zlim(0, 1)
        plt.show()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Given an output value from a neuron, we want to calculate it’s slope
    def sigmoid_prime(self, x):
        return x * (1 - x)

    def plot_error(self):
        plt.plot(self.error_x_y, self.error_z)
        plt.xlabel("Epoch")
        plt.ylabel("Mean squared error")
        plt.title("Mean squared error for each epoch (3 Layers)")
        plt.show()


print("Started at: " + str(dt.datetime.now()))
nn = NeuralNetwork()
# nn.create_dataset()
nn.create_network()
nn.start_training()
print("Ended at: " + str(dt.datetime.now()))
nn.plot_error()
nn.test_network()
