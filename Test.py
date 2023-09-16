import numpy as np
import pandas as pd
import json

with open("Dataset.json", "r") as file:
    data = json.load(file)

def ReLU(x):
    return np.maximum(x, 0)

def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a

test_data = pd.read_csv('Data/mnist_test.csv')
test_data = np.array(test_data)
m, n = test_data.shape
np.random.shuffle(test_data)

data_test = test_data[0:1].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255

m_, n_ = data_test.shape

weights_01 = data["Weights-01"]
biases_01 = data["Biases-01"]
weights_02 = data["Weights-02"]
biases_02 = data["Biases-02"]

weights_01 = np.array(weights_01)
biases_01 = np.array(biases_01)
weights_02 = np.array(weights_02)
biases_02 = np.array(biases_02)

z1 = weights_01.dot(X_test) + biases_01
a1 = ReLU(z1)
z2 = weights_02.dot(a1) + biases_02

data_test = data_test.T

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

max_val = z2[0]
index = 0
for i in range(len(z2)):
    if z2[i] >= max_val:
        index = i
        max_val = z2[i]

#print(z2)
print("I Think it is a ", index)