# Setup plotting
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)


import pandas as pd

red_wine = pd.read_csv('10_red-wine.csv')
print(red_wine.head())

print(red_wine.shape) # (rows, columns)

#The target is 'quality', and the remaining columns are the features.
#1) Input shape

input_shape = [11]


#2) Define a linear model
import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.Sequential([layers.Dense(units=1, input_shape=[11])])

#3) Look at the weights
#Internally, Keras represents the weights of a neural network with tensors. 
# Tensors are basically TensorFlow's version of a Numpy array with a few differences that 
# make them better suited to deep learning. One of the most important is that tensors are 
# compatible with GPU and TPU) accelerators. TPUs, in fact, are designed specifically
# for tensor computations.

w, b = model.weights

print("--->Weights:\n{}\n\n--->Bias:\n{}".format(w, b))


