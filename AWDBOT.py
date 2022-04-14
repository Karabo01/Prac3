import numpy as np
import csv
import time
import math
np.random.seed(0)

import sys

def progressbar(it, prefix="", size=60, file=sys.stdout):#Progress bar for indicating training progress
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "█"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


class Layer:# Class defines different layers in NN
    def __init__(self, num_inputs,num_neurons):#initializes random weights and Biases
        self.weights = 0.10*np.random.randn(num_inputs,num_neurons)
        self.biases = np.ones((1, num_neurons))

    def forwardPass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_RELU: # Activation class for neurons
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class softmax:
    def forward(self, inputs):#calculate the individual soft maxes of the nodes
        exp_values = np.exp(inputs * np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.ouput = probabilities

class loss:
    def getLoss(self, inputs, expected):
        fLoss = 0
        epoch = np.subtract(inputs,expected, out = None)
        epoch = np.square(epoch)
        print(epoch)
        for i in range(len(inputs)):
            for j in range(len(expected[0])):
                fLoss += epoch[i][j]
        self.output = 0.5 * fLoss

def reConvert(val):#Converts integers into RPS
    if val==1:
        return 'R'
    if val==2:
        return 'P'
    else:
        return 'S'

def convert(Char):#Converts RPS From Yt into desired Xt
    if Char=="R":
        return "P"
    if Char=="P":
        return "S"
    else:
        return "R"

def dataExtraction():# Extracts data from CSV File
    sequence_filename = "data1.csv"
    #for i in progressbar(range(100), "Computing: ", 100):
    with open(sequence_filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    print("Data extracted!!")
    return data#

def arrange(yt,xt):#Separates Yt and Xt into separate variables
    for i in range(len(data)):
        yt.append(str(data[i])[10:11])
        xt.append(str(data[i])[2:6])


def checkOneHot(Char):#Create onehot encoding
    if Char=='R':
        return 0
    elif Char=='P':
        return 1
    else:
        return 2

def getInputs(xt,percent):#Return a batch of inputs
    X=[]
    for i in range(int(percent*len(xt))):
        X.append([])
        for j in range(12):
            X[i].append(0)

    for i in range(int(percent*len(xt))):
        playX=str(xt[i][0:1])
        index=checkOneHot(playX)
        X[i][index] = 1
        playY=str(xt[i][1:2])
        index = checkOneHot(playY)
        X[i][index+3]=1

        playX = str(xt[i][2:3])
        index = checkOneHot(playX)
        X[i][index+6] = 1
        playY = str(xt[i][3:4])
        index = checkOneHot(playY)
        X[i][index + 9] = 1
    return X

def oneHotEncoding(arr, batch_size):
    onehot = []
    temp = [0,0,0]
    for i in range (batch_size):
        if arr[i] == 'R':
            temp[0] = 1
        elif arr[i] == 'P':
            temp[1] = 1
        elif arr[i] == 'S':
            temp[2] = 1
        onehot.append(temp)
        temp = [0,0,0]
    return onehot
data=dataExtraction()
yt=[]
xt=[]
p=1
percentageofCSV=p/1000000
arrange(yt,xt)
inputs=getInputs(xt,percentageofCSV)

inputLayer= Layer(12,12) #input layer creation
hiddenLayer= Layer(12,3)# Hidden layer creation
outputLayer= Layer(3,3)# Output layer

activation1=Activation_RELU() #activation for layer 1(Input layer)

inputLayer.forwardPass(inputs)

activation1.forward(inputLayer.output)#input layer goes into activation function in hidden layer
hiddenLayer.forwardPass(activation1.output)#result of activation goes into hidden feed forward

smActivation = softmax()
smActivation.forward(hiddenLayer.output)

print(smActivation.ouput)

exp = oneHotEncoding(yt,5) #five was just for tsting usually it's P
print(exp)
theLoss = loss()
theLoss.getLoss(smActivation.ouput,exp)
print(theLoss.output)
#We need need aa softmax before this activation before this
outputLayer.forwardPass(hiddenLayer.output)


print("hello")


