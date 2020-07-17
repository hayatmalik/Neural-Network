import numpy as np 
import sys
import random
import math

def getarr(index):
  array = np.zeros(10)
  array[index] = 1
  return array;

def readtarget(filename):
  file = np.loadtxt(filename, dtype='int')
  target = []
  for k in file:
    tg = getarr(k)
    target.append(tg)

  return target

def reading(filename):
  print("Reading from file")
  element = []
  data = []
  with open(filename,'r') as f:
    for line in f:
      for char in line.split():
        if(char[len(char) -1]==']'):
          char = char.replace(']','') 
          element.append(float(char))
          data.append(element)
          element = []
          continue
        if(char!='[' and char!=']' and char!=' '):
           	element.append(int(char))

  dataset = np.array(data)
  print("size of dataset is: " , len(dataset))
  return dataset

def Clustering(dataset):
  print("Clustering the dataset to determine means")
  k = 30
  i = len(dataset[0])
  # Get Random Centroids
  centres = []
  indices = np.random.randint(0,len(dataset)+1,k)  #indexes of the initial centroids
  for i in indices:
    centres.append(dataset[i])
  centroids = np.array(centres)   # an array of the vectors of 30 centroids
  #print("size of centroids is: " ,len(centroids) ,"of size: " ,len(centroids[0]))
  #clusters = np.zeros((30,1,784))
  clusters = [[] for l in range(30)]
  times = 0
  while (1):
    # Assign each point to a centroid: 
    for point in dataset:
      distance = []
      for c in centroids:
        d = np.linalg.norm(point - c)
        distance.append(d)                  # distance has euclidean distance corresponding to each centroid
      index = np.argmin(distance)           # number of the cluster with which it has minimum distance
      clusters[index].append(point)
      #clusters[index] = np.append(clusters[index],point)         # each point assigned a cluster

    # Get average for each cluster:
    updatedcentroids = []
    for B in clusters:
      average = np.mean(B, axis = 0)
      # finding closest point to the average
      avs = []
      for point in dataset:
        dist = np.linalg.norm(point - average)
        avs.append(dist)

      nearest = np.argmin(avs) 
      updatedcentroids.append(dataset[nearest])             # Get new centroid using the average point

    #Base case  
    newcentroids = np.array(updatedcentroids)
    if(np.array_equal(newcentroids,centroids)):             # if average not changing, break the loop
      centroids = newcentroids
      break;

    centroids = newcentroids                # Update the centroids array
    clusters = [[] for l in range(30)]      # empty the clusters array 
    #print(times)
    times = times + 1
    #LOOP STARTS AGAIN

  mean = np.array(centroids)
  print("Succesfully found " ,len(mean) , "means")
  return mean

def  neuralnetwork(dataset,mean,targets,weights,mode):
  sigma = 8
  #print("weights are" ,weights)
  for epoch in range(mode):
    print("epoch number: "  , epoch+1)
    correct = 0
    for it in range(len(dataset)):               # CHANGE THIS
      #Getting output from the hidden layer
      output = []
      inputt = np.array(dataset[it])                 # CHANGE THIS
      t = np.array(targets[it])
      for u in mean:
        m = np.subtract(inputt,u)/255
        #print("The length of m is: " ,len(m), "and m is: " ,m)
        square = np.dot(m,m.T)
        #print("The dot product equals: " ,square)
        power = (-1)*(square / (sigma*sigma))
        #print("The power equals: " ,power)
        out = np.exp(power)
        #print("The out equals: " ,out)
        output.append(out)
      #print("The " ,len(output) ,"outputs are" ,output)


      #Getting the weighted sums
      weightedsums = []
      for weight in weights:
        sum = np.dot(output,weight.T)
        weightedsums.append(sum)
      #print("The " ,len(weightedsums) ,"weightedsums are" ,weightedsums)

      #Activation Function (sigmoid)
      Probs = []
      for sums in weightedsums:
        denominator = 1 + np.exp((-1) *sums)
        p = 1/denominator
        Probs.append(p)
      y = np.array(Probs)
      #print("The final values are:" ,y)
      #print("The target value is:" ,t) 

      #Predictions and Accuracy:
      highest = np.argmax(y)
      real = np.argmax(t)
      if(highest==real):
        correct = correct+1

      #Weights Updation:
      newweights = np.zeros((10,30))
      eeta = 0.9
      if(mode == 2):
        eeta = sys.argv[4]                 
      for p in range(10):
        item = (-1) * (t[p]-y[p]) * y[p]*(1-y[p])*float(eeta)
        #print(item)
        for q in range(30):
          newweights[p][q] = weights[p][q] - (output[q] * item)
      weights = newweights
      #DataSet Loop Starts again
    print("Predicted correctly: " ,correct ,"out of " ,len(dataset))
    acc = (correct/len(dataset)) * 100
    print("Accuracy is:  " ,acc)

  return weights


filename = sys.argv[2]           
mode = sys.argv[1]                
labels = sys.argv[3]             
targets = readtarget(labels)  
dataset = reading(filename)         
mean = Clustering(dataset)
if(mode=="train"):                                              
  print("Training data in progress...")
  randomweights = np.random.uniform(-1.0, 1.0, size= (10,30))
  optimumweights = neuralnetwork(dataset,mean,targets,randomweights,2)
  # print("The Optimum Weights achieved: " ,optimumweights)
  np.savetxt('netweights.txt',optimumweights)


else:
  print("Testing in progress...")
  netweights = np.loadtxt(sys.argv[4])
  #netweights = np.loadtxt('netweights.txt')
  optimumweights = neuralnetwork(dataset,mean,targets,netweights,1)
