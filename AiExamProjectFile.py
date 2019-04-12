import math
import matplotlib.pyplot as plt
import numpy as np

# region Creating the dataset
############################
### Creating the dataset ###
############################

input = -1.0
x_list = []
y_list = []
# f = open("dataset.csv", "w")
# f.write("input,output\n")
# StringBuilder = ""

while input <= 1:
    x_list.append(input)
    output = round(math.sin(2 * math.pi * input) + math.sin(5 * math.pi * input), 3)
    y_list.append(output)
    # StringBuilder = StringBuilder + str(input) + "," + str(output) + "\n"
    input = round(input + 0.002, 3)

# f.write(StringBuilder)
#plt.plot(x_list, y_list)
#plt.axis([-1, 1, -2, 2])
#plt.show()
# endregion

###################################
### Creating the Neural Network ###
###################################

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

# Input/output data
X_input = np.array(x_list)
Y_input = np.array(y_list)

# Training (forward propagation through our network)
def forward(X):
    z = np.dot(X, W1)       # dot product of X (input) and first set of 1x3 weights
    print("z: " + str(z))
    z2 = sigmoid(z)         # activation function
    print("z2: " + str(z2))
    z3 = np.dot(z2, W2)     # dot product of hidden layer (z2) and second set of 3x1 weights
    print("z3: " + str(z3))
    prediction = sigmoid(z3)# final activation function
    return prediction

for index in range(0, x_list.__len__()):
    input = x_list[index]
    print("Input: " + str(input))
    predictedOutput = forward(x_list[index])
    print("Predicted output: " + str(predictedOutput))
    actualOutput = y_list[index]
    print("Actual output: " + str(actualOutput) + "\n")
