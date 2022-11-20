import tensorflow as tf

# Setup plotting
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)



#0) Inport Data 

import pandas as pd

concrete = pd.read_csv('20_concrete.csv')
print(concrete.head())


#1) Input Shape
#The target for this task is the column 'CompressiveStrength'. The remaining columns are the features we'll use as inputs.

input_shape = [8]

#2) Define a Model with Hidden Layers
# Now create a model with three hidden layers, each having 512 units and the ReLU activation.
# Be sure to include an output layer of one unit and no activation, and also input_shape 
# as an argument to the first layer.

from tensorflow import keras
from keras import layers

model =  keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=512, activation='relu', input_shape=[8]),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])


#3) Activation Layers

model = keras.Sequential([
    layers.Dense(32,  input_shape=[8]),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.Activation('relu'),
    layers.Dense(1),
])



#4)Alternatives to ReLU

#There is a whole family of variants of the 'relu' activation
# -- 'elu', 'selu', and 'swish', among others -- all of which you 
# can use in Keras. Sometimes one activation will perform better
# than another on a given task, so you could consider experimenting 
# with activations as you develop a model.
# The ReLU activation tends to do well on most problems, so it's a good one to start with.

# Change 'relu' to 'elu', 'selu', 'swish'... or something else
activation_layer = layers.Activation('swish')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()