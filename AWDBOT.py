import numpy as np
import csv
import time
import math
import matplotlib.pyplot as plt
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

class Sigmoid: # Activation class for neurons
    def forward(self, inputs):
        self.output = 1/(1+ np.exp((inputs - np.max(inputs, axis = 1, keepdims = True))))

class softmax:
    def forward(self, inputs):#calculate the individual soft maxes of the nodes
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities

class loss:
    def getLoss(self, inputs, expected):
        fLoss = 0
        epoch = np.subtract(expected,inputs, out = None)
        epoch = np.square(epoch)
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
            temp[1] = 1
        elif arr[i] == 'P':
            temp[2] = 1
        elif arr[i] == 'S':
            temp[0] = 1
        onehot.append(temp)
        temp = [0,0,0]

    output= np.array(onehot)
    return output

def Accuracy(inputs, expected):
    predicions_correct = inputs.argmax(axis=1) == expected.argmax(axis=1)
    accuracy = predicions_correct.mean()
    return accuracy


def backPropagation(rate, Olayer,Hlayer,Ilayer,expOut):
    outError= Olayer.output - expOut
    outDelta= outError * Olayer.output * (1 - Olayer.output)

    hiddenError= np.dot(outDelta, Olayer.weights)
    hiddenDelta = hiddenError * Hlayer.output * (1-Hlayer.output)

    newWeight1= np.dot(Hlayer.output.T,outDelta)/expOut.size
    newWeight2 = np.dot(Ilayer.output.T,hiddenDelta)/expOut.size

    Olayer.weights= Olayer.weights - rate* newWeight1
    Hlayer.weights = Hlayer.weights - rate* newWeight2


    filename = "input_hidden_updated.txt"
    with open(filename, 'w') as csvfile:  # Write data to text file
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(Hlayer.weights)

    filename = "hidden_output_updated.txt"
    with open(filename, 'w') as csvfile:  # Write data to text file
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(Olayer.weights)

def forwardPass(inputLayer,hiddenLayer,outputLayer,smActivation,activation1,theLoss,exp):

    inputLayer.forwardPass(inputs)

    activation1.forward(inputLayer.output)  # input layer goes into activation function in hidden layer

    hiddenLayer.forwardPass(activation1.output)  # result of activation goes into hidden feed forward

    activation1.forward(hiddenLayer.output)

    outputLayer.forwardPass(activation1.output)
    smActivation.forward(outputLayer.output)

    theLoss.getLoss(smActivation.output,exp)




data=dataExtraction()
yt=[]
xt=[]
p=900000
percentageofCSV=p/1000000
arrange(yt,xt)
inputs=getInputs(xt,percentageofCSV)

#Initialization
inputLayer= Layer(12,3) #input layer creation
hiddenLayer= Layer(3,3)# Hidden layer creation
outputLayer= Layer(3,3)# Output layer
smActivation = softmax()
activation1=Sigmoid() #activation for layer 1(Input layer)
theLoss = loss()
exp = oneHotEncoding(yt, p)


forwardPass(inputLayer,hiddenLayer,outputLayer,smActivation,activation1,theLoss,exp)

##Save weights to file
filename = "input_hidden.txt"
with open(filename, 'w') as csvfile:  # Write data to text file
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(hiddenLayer.weights)

filename = "hidden_output.txt"
with open(filename, 'w') as csvfile:  # Write data to text file
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(outputLayer.weights)


loss=[]
acc=[]
avgAcc=0
j=0
epochs=1
l = 0.1
while(avgAcc<0.8):
    for i in range(epochs):
        if(i==0):
            print("Training in progress: ")
        forwardPass(inputLayer,hiddenLayer,outputLayer,smActivation,activation1,theLoss,exp)
        loss.append(theLoss.output / p)
        acc.append(Accuracy(smActivation.output,exp))
        if (j / epochs >= l):
            #print("#", end=" ")
            print(theLoss.output / p)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
            l += 0.1
        j += 1

        backPropagation(0.2, outputLayer, hiddenLayer, inputLayer, exp)
        epochs+=1
    avgAcc=np.mean(acc)
    print("Mean accuracy: ", avgAcc)



#print(theLoss.output/p)
print(smActivation.output[:1])



forwardPass(inputLayer,hiddenLayer,outputLayer,smActivation,activation1,theLoss,exp)
print(smActivation.output[:1])

t=np.arange(0,epochs,1)
h=acc

plt.figure(1)
plt.plot(t,h)
plt.grid(True)
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy')
plt.title ('Accuracy')


t=np.arange(0,epochs,1)
h=loss
plt.figure(2)
plt.plot(t,h)
plt.grid(True)
plt.xlabel('# of Epochs')
plt.ylabel('Loss')
plt.title ('Loss')
plt.show()