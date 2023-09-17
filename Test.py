import numpy as np
import pandas as pd
import json

# Importing Stored Parameters from Json File
with open("Parameters.json", "r") as file:
    parameters = json.load(file)

# Activation Function Rectified Linear Unit (ReLU)
def ReLU(x):
    return np.maximum(x, 0)

# Activation Function Softmax
def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a

# Importing the Dataset for Testing
test_data = pd.read_csv('Data/mnist_test.csv')

# Turn Dataset into Numpy Array and Shuffling it
test_data = np.array(test_data)
m, n = test_data.shape
np.random.shuffle(test_data)

# Selecting one Digit from Shuffled Dataset
data_test = test_data[0:1].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255

# Extracting Weights and Biases from Json Lists
weights_01 = parameters["Weights-01"]
biases_01 = parameters["Biases-01"]
weights_02 = parameters["Weights-02"]
biases_02 = parameters["Biases-02"]

# Turning our Weights and Biases into Numpy Arrays
weights_01 = np.array(weights_01)
biases_01 = np.array(biases_01)
weights_02 = np.array(weights_02)
biases_02 = np.array(biases_02)

# Forward Propagating the Input
z1 = weights_01.dot(X_test) + biases_01
a1 = ReLU(z1)
z2 = weights_02.dot(a1) + biases_02
a2 = softmax(z2)

# Transposing the Dataset once again
data_test = data_test.T

# Drawing the Digit on the Console
for i in range(1, 785):
    if data_test[0][i] >= 0 and data_test[0][i] <= 30:
        print(" ", end=" ")
    elif data_test[0][i] >= 31 and data_test[0][i] <= 60:
        print(".", end=" ")
    elif data_test[0][i] >= 61 and data_test[0][i] <= 90:
        print(",", end=" ")
    elif data_test[0][i] >= 91 and data_test[0][i] <= 120:
        print("`", end=" ")
    elif data_test[0][i] >= 121 and data_test[0][i] <= 150:
        print("*", end=" ")
    elif data_test[0][i] >= 151 and data_test[0][i] <= 180:
        print("!", end=" ")
    elif data_test[0][i] >= 181 and data_test[0][i] <= 210:
        print("#", end=" ")
    elif data_test[0][i] >= 211:
        print("@", end=" ")
    if (i - 0) % 28 == 0:
        print("")

# Finding the index of Highest Probablity after Forward Propagation
max_val = a2[0]
index = 0
for i in range(len(a2)):
    if a2[i] >= max_val:
        index = i
        max_val = a2[i]

# Listing the Probablity of Each Digit being the Answer
for i in range(0, 10):
    print("Probablity of ", i, " is %.2f percent" % (a2[i][0] * 100))

# Printing the Prediction of our Neural Network
print("I Think it is a ", index)
