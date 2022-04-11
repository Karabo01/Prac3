import numpy as np
import csv
import time
import math
np.random.seed(0)

import sys

def progressbar(it, prefix="", size=60, file=sys.stdout):
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


class Layer:
    def _init_(self, num_inputs,num_neurons):
        self.weights = 0.10*np.random(num_inputs,num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def forwardPass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_RELU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



def reConvert(val):#Converts integers into RPS
    if val==1:
        return 'R'
    if val==2:
        return 'P'
    else:
        return 'S'

def convert(Char):#Converts RPS into integers
    if Char=="R":
        return "P"
    if Char=="P":
        return "S"
    else:
        return "R"

def dataExtraction():
    sequence_filename = "data1.csv"
    #for i in progressbar(range(100), "Computing: ", 100):
    with open(sequence_filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    print("Data extracted!!")
    return data

def arrange(yt,xt):
    for i in range(len(data)):
        yt.append(str(data[i])[10:11])
        xt.append(str(data[i])[2:6])

data=dataExtraction()
yt=[]
xt=[]
arrange(yt,xt)

print("hello")