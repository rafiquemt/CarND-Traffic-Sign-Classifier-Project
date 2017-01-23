"""Traffic sign classifier

Traffic sign classifier

"""
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'data/train.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

print(X_train.shape)

# ----------- Visualize
### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import random
import numpy as np
import matplotlib.pyplot as plt
import csv


import csv
sign_mapping = dict()
with open('./signnames.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    csvreader.__next__()
    for row in csvreader:
        sign_mapping[int(row[0])] = row[1] 
    
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

print(y_train[index], sign_mapping[y_train[index]])
