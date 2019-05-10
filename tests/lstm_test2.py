import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(7)

#load the dataset
dataframe = pandas.read_csv('../airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('flaot32')

# normalize the dataset betweet 0 and 1
mean = dataset.mean()
stddev = dataset.std()
dataset = (dataset - mean) / stddev, (mean, stddev) 

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))