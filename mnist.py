# Christopher Partridge
import numpy as np 

n = 20          #hidden units 
m = 0.9         #momentum
lrate = 0.1     #learning rate


# read in data sets
train_data = np.loadtxt("mnist_train.csv", delimiter=",")
test_data = np.loadtxt("mnist_test.csv", delimiter=",")

# store data in array of lists(matrix)


