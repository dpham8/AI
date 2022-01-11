from random import seed
from random import random
import math
import numpy as np
import csv

count = 0 # number of total data samples

hid_neurons = []
class hidden_neuron:
    def __init__(self, weight, theta, y = None):
        self.weight = weight
        self.theta = theta
        self.y = y
        hid_neurons.append(self)
        
out_neurons = []
class output_neuron:
     def __init__(self, weight, theta, delta = None, y = None, x = None):
        self.weight = weight
        self.theta = theta
        self.x = x
        self.y = y
        self.delta = delta
        out_neurons.append(self)

# input(x_vec) -> hidden layer (3 neurons) -> output layer (3 neurons)            
def feed_forward(x_vec):    
    global out_neurons
    global hid_neurons
         
    x_vec = np.array(x_vec, dtype = float)
    # calculating hidden layer and pushing new x inputs to the output layer
    for i in range(3): 
        vec = []
        for neuron in hid_neurons:
            x = np.dot(x_vec,neuron.weight) - neuron.theta
            y = 1/(1 + math.exp(-1 * x))
            neuron.y = y 
            vec.append(y)    
        out_neurons[i].x = vec 
    
    # calculating output layer
    for neuron in out_neurons:
        x = np.dot(neuron.x,neuron.weight) - neuron.theta
        y = 1/(1 + math.exp(-1 * x))
        neuron.y = y

def backward_chaining(x_vec, yd_vec):
    global out_neurons
    global hid_neurons       
    alpha = 0.1
    bias = -1
    x_vec = np.array(x_vec, dtype = float)
    yd_vec = np.array(yd_vec, dtype = float)
    
    # backpropagation for output neurons
    i = 0 
    for i in range(3):
        if out_neurons[i].y != yd_vec[0]:
            delta = out_neurons[i].y * (1 - out_neurons[i].y) * yd_vec[i]
            out_neurons[i].x = np.array(out_neurons[i].x, dtype = float) 
            w = out_neurons[i].weight + alpha * out_neurons[i].x * delta
            theta = out_neurons[i].theta + alpha * bias * delta 
            output_neuron(w,theta,delta,out_neurons[i].y)
    del out_neurons[:3]

    # backpropagation for hidden neurons
    i = 0
    for i in range(3):
        summation = 0
        j = 0
        for k in range(len(out_neurons)):
            summation = summation + out_neurons[k].weight[j] * out_neurons[k].delta 
        j += 1

        delta = hid_neurons[i].y * (1 - hid_neurons[i].y) * summation
        w = hid_neurons[i].weight + alpha * x_vec * delta
        theta = hid_neurons[i].theta + alpha * bias * delta
        hidden_neuron(w,theta)        
    del hid_neurons[:3]

def performance(yd_vec):
    global out_neurons
    output = 0
    i = 0  
    for neuron in out_neurons:
        n = abs(yd_vec[i] - neuron.y)
        output += n
        i += 1
    return out

FILENAME = 'iris.csv'
with open(FILENAME, newline='') as csvfile:
    reader = csv.DictReader(csvfile) 
    data = []
    for row in csv.reader( csvfile, delimiter =','):
        count+=1
        data.append(row)

# generating random hidden neuron weights
seed(0) 
for i in range(3):
    w = [random() for i in range(len(data[0])-3)] # subtract 3 because output is a 3-vector
    theta = random()
    w = np.array(w,dtype = float)
    hidden_neuron(w,theta)

seed(1)
# generating random output neuron weights
for i in range(3):
    w = [random() for i in range(3)]
    theta = random()
    w = np.array(w,dtype = float)
    output_neuron(w,theta)

for epoch in range(50):
    summation = 0 
    iteration = 1
    for row in data:
        feed_forward(row[:4])
        backward_chaining(row[:4],row[4:])
        
        # caculating ||yd - y||
        i = 0
        output = 0  
        yd_vec = row[4:]
        yd_vec = np.array(yd_vec, dtype = float)
        for neuron in out_neurons:
            n = abs(yd_vec[i] - neuron.y)
            output += n
            i += 1
        summation += output
        print("Epoch " + str(epoch+1) + ", Iteration" + str(iteration) + ", Prediction is [" + "{:.6f}".format(out_neurons[0].y) + "," + "{:.6f}".format(out_neurons[1].y) + "," + "{:.6f}".format(out_neurons[2].y))
        iteration += 1
    performance = summation/count
    print("Epoch" + str(epoch+1) + ", Results: performance = " + "{:.6f}".format(performance))
        
    

    
  

