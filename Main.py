import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

def init_parameters():
    weights_01 = np.random.rand(10, 784) - 0.5
    biases_01 = np.random.rand(10, 1) - 0.5
    weights_02 = np.random.rand(10, 10) - 0.5
    biases_02 = np.random.rand(10, 1) - 0.5
    return weights_01, biases_01, weights_02, biases_02

def ReLU(x):
    return np.maximum(x, 0)

def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a

def forward_propagation(weights_01, biases_01, weights_02, biases_02, x):
    z1 = weights_01.dot(x) + biases_01
    a1 = ReLU(z1)
    z2 = weights_02.dot(a1) + biases_02
    a2 = softmax(z2)
    return z1, a1, z2, a2

def ReLU_derivative(x):
    return x > 0

def one_hot(y):
    one_hot_y = np.zeros((Y_train.size, Y_train.max() + 1))
    one_hot_y[np.arange(Y_train.size), Y_train] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def back_propagation(z1, a1, z2, a2, weights_01, weights_02, x, y):
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dweights_02 = 1 / m * dz2.dot(a1.T)
    dbiases_02 = 1 / m * np.sum(dz2)
    dz1 = weights_02.T.dot(dz2) * ReLU_derivative(z1)
    dweights_01 = 1 / m * dz1.dot(x.T)
    dbiases_01 = 1 / m * np.sum(dz1)
    return dweights_01, dbiases_01, dweights_02, dbiases_02

def update_parameters(weights_01, biases_01, weights_02, biases_02, dweights_01, dbiases_01, dweights_02, dbiases_02, alpha):
    weights_01 = weights_01 - alpha * dweights_01
    biases_01 = biases_01 - alpha * dbiases_01
    weights_02 = weights_02 - alpha * dweights_02
    biases_02 = biases_02 - alpha * dbiases_02
    return weights_01, biases_01, weights_02, biases_02

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

accuracy_list = []
iteration_list = []

def gradient_descent(x, y, iterations, alpha):
    weights_01, biases_01, weights_02, biases_02 = init_parameters()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(weights_01, biases_01, weights_02, biases_02, x)
        dweights_01, dbiases_01, dweights_02, dbiases_02 = back_propagation(z1, a1, z2, a2, weights_01, weights_02, x, y)
        weights_01, biases_01, weights_02, biases_02 = update_parameters(weights_01, biases_01, weights_02, biases_02, dweights_01, dbiases_01, dweights_02, dbiases_02, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(a2), y))
            iteration_list.append(i)
            accuracy_list.append(get_accuracy(get_predictions(a2), y) * 100)
    return weights_01, biases_01, weights_02, biases_02

weights_01, biases_01, weights_02, biases_02 = gradient_descent(X_train, Y_train, 1000, 0.1)

# Plotting the Accuracy of Neural Network
plt.plot(iteration_list, accuracy_list, color='g')
plt.axis([0, 1000, 0, 100])
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.show()