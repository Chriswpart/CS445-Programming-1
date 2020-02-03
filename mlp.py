# Christopher Partridge
import csv
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import confusion_matrix, accuracy_score

# experiment parameters
bias = 1            #bias
n = 100              #hidden units
alpha = 0.9         #momentum thingy
lrate = 0.1         #learning rate
ep = 50             #epochs

# read in data sets
f = open("mnist_train.csv","r")
data1 = csv.reader(f)
train = np.array(list(data1))
f1 = open("mnist_test.csv","r")
data2 = csv.reader(f1)
test = np.array(list(data2))

# store data and labels
# scale data between 0 and 1
train_data = np.asfarray(train[:, 1:]) / 255
#temp = np.asfarray(train_data[:15000])
temp = np.asfarray(train_data[:30000])
train_data = temp
test_data = np.asfarray(test[:, 1:]) / 255
train_label = np.asfarray(train[:, :1])
test_label = np.asfarray(test[:, :1])

# used for confusion matrix
cm = np.zeros((10,10),dtype=int)

# one hot .9 and .1
lr = np.arange(10)
train_oh = (lr == train_label).astype(np.float)
test_oh = (lr == test_label).astype(np.float)
train_oh[train_oh == 0] = 0.1
train_oh[train_oh == 1] = 0.9
test_oh[test_oh == 0] = 0.1
test_oh[test_oh == 1] = 0.9

# weights
wih = np.random.uniform(-0.05, 0.05,(784, n))
woh = np.random.uniform(-0.05, 0.05,(n + 1, 10))
prevwoh = np.zeros((n+1, 10))
prevwih = np.zeros((784, n))

#activations
hi = np.zeros((1, n+1))
hi[0, 0] = 1


# multilayer perceptron
def mlp(epoch, data, label, one_hot, flag):
  global wih, woh, prevwih, prevwoh, cm, ep
  predictions = []     # store predictions
  actuals = []         # store actual targets

  # loops through training or testing data
  for i in range(data.shape[0]):
    target = label[i, 0].astype('int')    # get target value
    actuals.append(target)                 # store in actual list
    x = data[i]                           # grab data
    x[0] = bias                           # set bias
    x = x.reshape(1, 784)                 # reshape x

    # activation of hidden and output layers
    zh = np.dot(x,wih)
    sigh = expit(zh)
    hi[0,1:] = sigh
    zo = np.dot(hi, woh)
    sigo = expit(zo)

    # find argmax and store prediction
    predict = np.argmax(sigo)
    predictions.append(predict)

    # train perceptrons
    if epoch > 0 and flag == 1:
      # error for output
      erroro = sigo * (1 - sigo) * (one_hot[i] - sigo)
      # error for hidden layers
      errorh = sigh * ( 1- sigh) * np.dot(erroro, woh[1:,:].T)

      # update output weights
      deltawoh = (lrate * erroro * hi.T) + (alpha * prevwoh)
      prevwoh = deltawoh
      woh = woh + deltawoh
      # update hidden layer weights
      deltawih = (lrate * errorh * x.T) + (alpha * prevwih)
      prevwih = deltawih
      wih = wih + deltawih

  # calculate accuracy
  accuracy = (np.array(predictions) == np.array(actuals)).sum()/float(len(actuals))*100


  # only build confusion matrix after test data
  if(flag == 0):
    tmp = confusion_matrix(actuals, predictions)
    cm = np.add(tmp, cm)

  # print confusion matrix
  if(flag == 0 and epoch == (ep - 1)):
    print("Confusion Matrix For Epoch")
    print(cm)
    #print(confusion_matrix(actuals, predictions))

  return accuracy

# store accuracy into file
def store_accuracy(accur, accuracy, data):
  with open(data, 'a') as myfile:
    wr = csv.writer(myfile)
    wr.writerow([accur, accuracy])


# build graph of epoch and accuracy
def build_graph():
  x1, y1 = np.loadtxt("train_accur.csv",delimiter=',',unpack=True)
  x2, y2 = np.loadtxt("test_accur.csv",delimiter=',',unpack=True)
  plt.plot(x1,y1, label="Training Set")
  plt.plot(x2,y2, label="Testing Set")
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy ')
  plt.title('# Hidden Units: ' + str(n) + ' Alpha: ' + str(alpha))
  plt.legend()
  plt.show()
  plt.savefig('accurepoch.png')


# vaules used in experiment
print('Hidden nodes: ' + str(n) + '. Alpha: ' + str(alpha) + '. Training set: ' + str(len(train_data)) + '. Test set: ' + str(len(test_data)))

# train and test for each epoch
for each in range(ep):
  train_accur = mlp(each, train_data, train_label, train_oh, 1)
  test_accur = mlp(each, test_data, test_label, test_oh, 0)
  store_accuracy(each, train_accur, 'train_accur.csv')
  store_accuracy(each, test_accur, 'test_accur.csv')

# build graph from accuracy data
build_graph()
