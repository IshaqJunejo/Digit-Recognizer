import numpy as np
import pandas as pd
#from matplotlib import pylpot as plt

data = pd.read_csv('Data/mnist_train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255

def ReLU(x):
    return np.maximum(x, 0)

def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a

def ReLU_derivative(x):
    return x > 0

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def init_parameters():
    weights_01 = np.random.rand(16, 784) - 0.5
    biases_01 = np.random.rand(16, 1) - 0.5
    weights_02 = np.random.rand(16, 16) - 0.5
    biases_02 = np.random.rand(16, 1) - 0.5
    weights_03 = np.random.rand(10, 16) - 0.5
    biases_03 = np.random.rand(10, 1) - 0.5
    return weights_01, biases_01, weights_02, biases_02, weights_03, biases_03

def forward_propagation(weights_01, biases_01, weights_02, biases_02, weights_03, biases_03, x):
    z1 = weights_01.dot(x) + biases_01
    a1 = ReLU(z1)
    z2 = weights_02.dot(a1) + biases_02
    a2 = ReLU(z2)
    z3 = weights_03.dot(a2) + biases_03
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def back_propagation(z1, a1, z2, a2, z3, a3, weights_01, weights_02, weights_03, x, y):
    one_hot_y = one_hot(y)
    dz3 = a3 - one_hot_y
    dweights_03 = 1 / m * dz3.dot(a2.T)
    dbiases_03 = 1 / m * np.sum(dz3)
    dz2 = weights_03.T.dot(dz3) * ReLU_derivative(z2)
    dweights_02 = 1 / m * dz2.dot(a1.T)
    dbiases_02 = 1 / m * np.sum(dz2)
    dz1 = weights_02.T.dot(dz2) * ReLU_derivative(z1)
    dweights_01 = 1 / m * dz1.dot(x.T)
    dbiases_01 = 1 / m * np.sum(dz1)
    return dweights_01, dbiases_01, dweights_02, dbiases_02, dweights_03, dbiases_03

def update_parameters(weights_01, biases_01, weights_02, biases_02, weights_03, biases_03, dweights_01, dbiases_01, dweights_02, dbiases_02, dweights_03, dbiases_03, alpha):
    weights_01 = weights_01 - (dweights_01 * alpha)
    biases_01 = biases_01 - (dbiases_01 * alpha)
    weights_02 = weights_02 - (dweights_02 * alpha)
    biases_02 = biases_02 - (dbiases_02 * alpha)
    weights_03 = weights_03 - (dweights_03 * alpha)
    biases_03 = biases_03 - (dbiases_03 * alpha)
    return weights_01, biases_01, weights_02, biases_02, weights_03, biases_03

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, iterations, alpha):
    weights_01, biases_01, weights_02, biases_02, weights_03, biases_03 = init_parameters()
    for i in range(iterations + 1):
        z1, a1, z2, a2, z3, a3 = forward_propagation(weights_01, biases_01, weights_02, biases_02, weights_03, biases_03, x)
        dweights_01, dbiases_01, dweights_02, dbiases_02, dweights_03, dbiases_03 = back_propagation(z1, a1, z2, a2, z3, a3, weights_01, weights_02, weights_03, x, y)
        weights_01, biases_01, weights_02, biases_02, weights_03, biases_03 = update_parameters(weights_01, biases_01, weights_02, biases_02, weights_03, biases_03, dweights_01, dbiases_01, dweights_02, dbiases_02, dweights_03, dbiases_03, alpha)
        if i % 200 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(a2), Y_train) * 100, "%")
    
    return weights_01, biases_01, weights_02, biases_02, weights_03, biases_03

weights_01, biases_01, weights_02, biases_02, weights_03, biases_03 = gradient_descent(X_train, Y_train, 8000, 0.02)
