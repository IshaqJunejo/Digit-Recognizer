import numpy as np
import pandas as pd
import random

# Importing the Dataset
data = pd.read_csv('Data/mnist_train.csv')

# Turning Dataset into Numpy Array
data = np.array(data)

# Running a While Loop
run = 1
while run:
    # Selecting a Random Number as Index
    index = random.randrange(0, 4997)

    # Drawing the Digit of that Index on Console
    for i in range(1, 785):
        if data[index][i] >= 0 and data[index][i] <= 30:
            print(" ", end=" ")
        elif data[index][i] >= 31 and data[index][i] <= 60:
            print(".", end=" ")
        elif data[index][i] >= 61 and data[index][i] <= 90:
            print(",", end=" ")
        elif data[index][i] >= 91 and data[index][i] <= 120:
            print("`", end=" ")
        elif data[index][i] >= 121 and data[index][i] <= 150:
            print("*", end=" ")
        elif data[index][i] >= 151 and data[index][i] <= 180:
            print("!", end=" ")
        elif data[index][i] >= 181 and data[index][i] <= 210:
            print("#", end=" ")
        elif data[index][i] >= 211:
            print("@", end=" ")
        if (i - 0) % 28 == 0:
            print("")
    
    # Print Info about that Digit
    print("Index of this Image is: ", index)
    print("It is Labeled as: ", data[index][0])
    print("")
    # Ask the User if He Wants To Try Again
    print("Wanna Try it Again?")
    print("0 = No")
    print("1 = Yes")
    run = int(input())
