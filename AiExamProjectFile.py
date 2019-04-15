import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# region Create dataset
# input = -1.0
# x_list = []
# y_list = []
# f = open("dataset.csv", "w")
# f.write("input,output\n")
# StringBuilder = ""
#
# while input <= 1:
#     x_list.append(input)
#     output = round(math.sin(2 * math.pi * input) + math.sin(5 * math.pi * input), 3)
#     y_list.append(output)
#     StringBuilder = StringBuilder + str(input) + "," + str(output) + "\n"
#     input = round(input + 0.002, 3)
#
# f.write(StringBuilder)
# plt.plot(x_list, y_list)
# plt.axis([-1, 1, -2, 2])
# plt.show()
# endregion

# region Helper methods
# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid function
def sigmoid_prime(input_value):
    return input_value * (1 - input_value)


# endregion

w1 = None
w2 = None
x_input = None
y_input = None
output_at_hidden_layer = None


def create_network():
    global w1
    global w2
    global x_input
    global y_input

    # Import data
    data_from_csv = pd.read_csv('dataset.csv')
    x_input = np.array(data_from_csv["input"])
    y_input = np.array(data_from_csv["output"])

    # Network structure
    input_size = 1
    output_size = 1
    hidden_size = 3

    # Weights
    np.random.seed(1)
    w1 = np.random.randn(input_size, hidden_size)  # (1x3) weight matrix from input to hidden layer
    print("W1: " + str(w1))
    w2 = np.random.randn(hidden_size, output_size)  # (3x1) weight matrix from hidden to output layer
    print("W2: " + str(w2))

    start_training()


def start_training():
    global x_input
    global y_input

    # for index in range(0, x_input.size):
    for index in range(0, 2):
        input_value = x_input[index]
        print("Input: " + str(input_value))
        output_value = forward_propagation(x_input[index])
        print("Predicted output: " + str(output_value))
        actual_output = sigmoid(y_input[index])
        print("Actual output: " + str(actual_output))
        back_propagation(input_value, actual_output, output_value)


def forward_propagation(input_value):
    global w1
    global w2
    global output_at_hidden_layer

    z = np.dot(input_value, w1)  # dot product of input_value and first set of weights (1x3)
    print("Input · w1 = z -> " + str(z))
    z2 = sigmoid(z)  # insert dot product z into activation function
    output_at_hidden_layer = z2
    print("Run z through sigmoid = z2 -> " + str(z2))
    z3 = np.dot(z2, w2)  # dot product of hidden layer (z2) and second set of 3x1 weights
    print("z2 · w2 = z3 -> " + str(z3))
    prediction = sigmoid(z3)  # final activation function
    print("Run z3 through sigmoid = prediction -> " + str(prediction))
    return prediction


def back_propagation(input_value, actual_output, output_at_output_layer):
    global w1
    global w2
    global output_at_hidden_layer

    layer2_error = actual_output - output_at_output_layer
    layer2_delta = layer2_error * sigmoid_prime(output_at_output_layer)
    layer1_error = np.dot(layer2_delta, w2.T)
    layer1_delta = layer1_error * sigmoid_prime(output_at_hidden_layer)

    # Update weights
    w2 = np.dot(output_at_hidden_layer.T, layer2_delta)
    w1 = np.dot(input_value.T, layer1_delta)

    print("Error: " + str(layer2_error))


create_network()
