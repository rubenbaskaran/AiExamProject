import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# region Creating the dataset
############################
### Creating the dataset ###
############################

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

###################################
### Creating the Neural Network ###
###################################

# Input/output data
dataFromCsv = pd.read_csv('dataset.csv')
X_input = np.array(dataFromCsv["input"])
Y_input = np.array(dataFromCsv["output"])

# Network structure
inputSize = 1
outputSize = 1
hiddenSize = 3

# Weights
np.random.seed(1)
W1 = np.random.randn(inputSize, hiddenSize)  # (1x3) weight matrix from input to hidden layer
print("W1: " + str(W1))
W2 = np.random.randn(hiddenSize, outputSize)  # (3x1) weight matrix from hidden to output layer
print("W2: " + str(W2))


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid function
def sigmoid_prime(input):
    return input * (1 - input)


# Training (forward propagation through our network)
def forward(input):
    global output_at_input_layer
    global output_at_hidden_layer

    z = np.dot(input, W1)  # dot product of input and first set of weights (1x3)
    output_at_input_layer = z
    print("Input · W1 = z -> " + str(z))
    z2 = sigmoid(z)  # insert dot product z into activation function
    output_at_hidden_layer = z2
    print("Run z through sigmoid = z2 -> " + str(z2))
    z3 = np.dot(z2, W2)  # dot product of hidden layer (z2) and second set of 3x1 weights
    print("z2 · W2 = z3 -> " + str(z3))
    prediction = sigmoid(z3)  # final activation function
    print("Run z3 through sigmoid = prediction -> " + str(prediction))
    return prediction


def errorfunction(actual_output, predicted_output):
    return actual_output - predicted_output


# for index in range(0, X_input.size):
for index in range(1, X_input.size):
    input = X_input[index]
    print("Input: " + str(input))
    output_at_output_layer = forward(X_input[index])
    print("Predicted output: " + str(output_at_output_layer))
    actualOutput = sigmoid(Y_input[index])
    print("Actual output: " + str(actualOutput))

    layer2_error = errorfunction(actualOutput, output_at_output_layer)
    layer2_delta = layer2_error * sigmoid_prime(output_at_output_layer)
    layer1_error = np.dot(layer2_delta, W2.T)
    layer1_delta = layer1_error * sigmoid_prime(output_at_hidden_layer)

    # Update weights
    W2 = np.dot(output_at_hidden_layer.T, layer2_delta)
    W1 = np.dot(input.T, layer1_delta)

    print("Error: " + str(layer2_error))
