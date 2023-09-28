import numpy as np
import pandas as pd
import json

def import_parameters(file_dir):
    # Importing Stored Parameters from Json File
    with open(file_dir, "r") as file:
        parameters = json.load(file)

    # Extracting Weights and Biases from Json Lists
    w1 = parameters["Weights-01"]
    b1 = parameters["Biases-01"]
    w2 = parameters["Weights-02"]
    b2 = parameters["Biases-02"]

    # Turning our Weights and Biases into Numpy Arrays
    w1 = np.array(w1)
    b1 = np.array(b1)
    w2 = np.array(w2)
    b2 = np.array(b2)

    return w1, b1, w2, b2

def import_dataset(file_dir):
    # Importing the Dataset for Testing
    test_data = pd.read_csv(file_dir)

    # Turn Dataset into Numpy Array and Shuffling it
    test_data = np.array(test_data)
    m, n = test_data.shape
    np.random.shuffle(test_data)

    # Selecting one Digit from Shuffled Dataset
    data = test_data[0:1].T
    y = data[0]
    x = data[1:n]
    x = x / 255

    return data, x, y

# Activation Function Rectified Linear Unit (ReLU)
def ReLU(x):
    return np.maximum(x, 0)

# Activation Function Softmax
def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a

def forward_propagation(w1, b1, w2, b2, x):
    # Forward Propagating the Input
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)

    return a2

def draw_digit(data):
    # Transposing the Dataset once again
    data = data.T

    # Drawing the Digit on the Console
    for i in range(1, 785):
        if data[0][i] >= 0 and data[0][i] <= 30:
            print(" ", end=" ")
        elif data[0][i] >= 31 and data[0][i] <= 60:
            print(".", end=" ")
        elif data[0][i] >= 61 and data[0][i] <= 90:
            print(",", end=" ")
        elif data[0][i] >= 91 and data[0][i] <= 120:
            print("`", end=" ")
        elif data[0][i] >= 121 and data[0][i] <= 150:
            print("*", end=" ")
        elif data[0][i] >= 151 and data[0][i] <= 180:
            print("!", end=" ")
        elif data[0][i] >= 181 and data[0][i] <= 210:
            print("#", end=" ")
        elif data[0][i] >= 211:
            print("@", end=" ")
        if (i - 0) % 28 == 0:
            print(" ", end="\n")

def print_ans(a2):
    # Listing the Probablity of Each Digit being the Answer
    for i in range(0, 10):
        print("Probablity of ", i, " is %.2f percent" % (a2[i][0] * 100))
    
    # Finding the index of Highest Probablity after Forward Propagation
    max_val = a2[0]
    max_val_index = 0
    for i in range(len(a2)):
        if a2[i] >= max_val:
            max_val_index = i
            max_val = a2[i]

    # Printing the Prediction of our Neural Network
    print("\nI Think it is a ", max_val_index, "\n")

# Executing the Test
weights_01, biases_01, weights_02, biases_02 = import_parameters("Parameters.json")
data_test, X_test, Y_test = import_dataset("Data/mnist_test.csv")
probabilites = forward_propagation(weights_01, biases_01, weights_02, biases_02, X_test)
draw_digit(data_test)
print_ans(probabilites)