import numpy as np
import csv
import time
import math
import matplotlib.pyplot as plt
np.random.seed(0)




class Layer:# Class defines different layers in NN
    def __init__(self, num_inputs,num_neurons):#initializes random weights and Biases
        self.weights = 0.10*np.random.randn(num_inputs,num_neurons)
        self.biases = np.ones((1, num_neurons))

    def forwardPass(self, X):
        inputs=np.array(X)
        self.output = np.dot(inputs, self.weights) + self.biases

class Sigmoid: # Activation class for neurons
    def forward(self, inputs):
        self.output = 1/(1+ np.exp(-(inputs)))

class softmax:
    def forward(self, inputs):#calculate the individual soft maxes of the nodes
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities

class loss:
    def getLoss(self, inputs, expected): # returns Loss
        fLoss = 0
        epoch = np.subtract(expected,inputs, out = None)
        epoch = np.square(epoch)
        for i in range(len(inputs)):
            for j in range(len(expected[0])):
                fLoss += epoch[i][j]
        self.output = 0.5 * fLoss

    def getMSE(self,inputs, expected):
        N = expected.size
        mse = ((inputs - expected) ** 2).sum() / (2 * N)
        self.output = mse

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
    return data

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

def oneHotEncoding(arr, batch_size): # one hot encodes RPS
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

def Accuracy(inputs, expected):# Calculates the accuarcy of the predicted output
    predicions_correct = inputs.argmax(axis=1) == expected.argmax(axis=1)
    accuracy = predicions_correct.mean()
    return accuracy

def importWeights(HLayer, OLayer):# imports optial weights saved from training
    inLayer_filename = "input_hidden_opt.npy"

    print("Input layer weights imported!!")
    data = np.load(inLayer_filename)
    HLayer.weights=np.array(data)

    oLayer_filename = "hidden_output_opt.npy"
    data = np.load(oLayer_filename)
    print("Hidden layer weights extracted!!")
    OLayer.weights = np.array(data)

def backPropagation(rate, Olayer,Hlayer,expOut, activation1, smActivation):
    X=np.array(inputs)
    outError= smActivation.output - expOut
    outDelta= outError * smActivation.output * (1 - smActivation.output)

    hiddenError= np.dot(outDelta, Olayer.weights.T)
    hiddenDelta = hiddenError * activation1.output * (1-activation1.output)

    newWeightH= np.dot(activation1.output.T,outDelta)/expOut.size
    newWeightI = np.dot(X.T,hiddenDelta)/expOut.size

    Olayer.weights= Olayer.weights - rate* newWeightH
    Hlayer.weights = Hlayer.weights - rate*newWeightI

    filename = "input_hidden_updated-Copy.npy"
    np.save(filename, Hlayer.weights)
    with open("input_hidden_updated.txt", 'w') as csvfile:  # Write data to text file
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(Hlayer.weights)

    filename = "hidden_output_updated-Copy.npy"
    np.save(filename, Olayer.weights)
    with open("hidden_output_updated.txt", 'w') as csvfile:  # Write data to text file
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(Olayer.weights)


def forwardPass(X,hiddenLayer,outputLayer,smActivation,activation1,theLoss,exp): # Does a forward propagation

    hiddenLayer.forwardPass(X)  # result of activation goes into hidden feed forward

    activation1.forward(hiddenLayer.output)

    outputLayer.forwardPass(activation1.output)
    smActivation.forward(outputLayer.output)

    theLoss.getLoss(smActivation.output,exp)


def testPhase(xt,yt):# Imports testing data and tests
    inputsTest=getInputs(xt[700000:],1)
    exp = oneHotEncoding(yt[700000:], 300000)
    forwardPass(inputsTest,hiddenLayer,outputLayer,activation2,activation1,theLoss,exp)
    print("Prediction: ",activation2.output)
    print("Expected: ",exp)

def saveWeights(Hlayer,Olayer):# saves optimal weights in file
    filename = "input_hidden_opt.npy"
    np.save(filename, Hlayer.weights)

    filename = "hidden_output_opt.npy"
    np.save(filename, Olayer.weights)

data=dataExtraction()
yt=[]
xt=[]
p=700000
percentageofCSV=p/1000000
arrange(yt,xt)
inputs=getInputs(xt,percentageofCSV)

#Initialization
hiddenLayer= Layer(12,3)# Hidden layer creation
outputLayer= Layer(3,3)# Output layer
smActivation = softmax()
activation1=Sigmoid() #activation for layer 1
activation2=Sigmoid()# acticavtion of output layer
theLoss = loss()
exp = oneHotEncoding(yt, p)

#importWeights(hiddenLayer,outputLayer)
forwardPass(inputs,hiddenLayer,outputLayer,activation2,activation1,theLoss,exp) # First forward pass to initialize network

##Save weights to file
filename = "input_hidden.npy"
np.save(filename, hiddenLayer.weights)
with open("input_hidden.txt", 'w') as csvfile:  # Write data to text file
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(hiddenLayer.weights)

filename = "hidden_output.npy"
np.save(filename, outputLayer.weights)
with open("hidden_output.txt", 'w') as csvfile:  # Write data to text file
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(outputLayer.weights)


loss=[]
acc=[]
avgAcc=0
j=0
epochs = 58
l = 0.1
rate=0.8
b=0
#importWeights(hiddenLayer,outputLayer)
while(avgAcc<0.01):# repeats training process until certain accuracy is achieved
    for i in range(epochs):
        if(i==0):
            print("Training in progress: ")
        forwardPass(inputs,hiddenLayer,outputLayer,activation2,activation1,theLoss,exp)
        loss.append(theLoss.output/p)
        acc.append(Accuracy(activation2.output,exp))
        if (j / epochs >= l):
            print(theLoss.output/p)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
            l += 0.1
        j += 1
        if(acc[len(acc)-1]>acc[len(acc)-2]>0.46):
            print("Saving Weights")
            saveWeights(hiddenLayer,outputLayer)
            rate=-0.2
        backPropagation(rate, outputLayer, hiddenLayer, exp, activation2,activation1)
    avgAcc=np.mean(acc)
    print("Mean accuracy: ", avgAcc)





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

time.sleep(2)
importWeights(hiddenLayer,outputLayer)
print("Testing phase begin.....", "Good luck")
testPhase(xt,yt)

