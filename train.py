import os
import numpy as np

import tensorflow as tf

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

g_train = []
l_train = []
s_train = []
y_train = []

file_path = "C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/processed/"

files = os.listdir(file_path)
files.sort()

length = len(files)
count = 0
for file in files:
    if count%100 == 0:
        print(str(count) + "/" + str(length))
    count += 1

    if file.endswith("g.npy"):
        g_train.append(np.subtract(np.load(file_path + file), 1.0))
    elif file.endswith("l.npy"):
        l_train.append(np.subtract(np.load(file_path + file), 1.0))
    elif file.endswith("s.npy"):
        s_train.append(np.load(file_path + file))
    elif file.endswith("y.npy"):
        y_train.append(np.load(file_path + file))
    else:
        pass

seed = 42

import random
random.Random(seed).shuffle(l_train)
random.Random(seed).shuffle(g_train)
random.Random(seed).shuffle(s_train)
random.Random(seed).shuffle(y_train)

size = 1515

l_train = l_train[0:size]
g_train = g_train[0:size]
s_train = s_train[0:size]
y_train = y_train[0:size]

print(len(l_train))
print(len(g_train))
print(len(s_train))
print(len(y_train))

l_train = np.array(l_train)
g_train = np.array(g_train)
s_train = np.array(s_train) 
y_train = np.array(y_train) 

l_train[np.isnan(l_train)] = 0
g_train[np.isnan(g_train)] = 0
s_train[np.isnan(s_train)] = 0
y_train[np.isnan(y_train)] = 0

print(l_train[0])

import keras
from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPool1D, Input, concatenate, Flatten
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, SGD

# define two sets of inputs
local_input = Input(shape=(100,1)) # 100 bins
global_input = Input(shape=(1000,1)) # 1000 bins

conv_act = "relu"
dense_act = "relu"
softmax_act = "relu"

# local branch
l = Conv1D(16, 5, activation=conv_act)(local_input)
l = Conv1D(16, 5, activation=conv_act)(local_input)
l = MaxPool1D(7, 2)(l)
l = Conv1D(32, 5, activation=conv_act)(local_input)
l = Conv1D(32, 5, activation=conv_act)(local_input)
l = MaxPool1D(7, 2)(l)
l = Flatten()(l)
l = Model(inputs=local_input, outputs=l)

# global branch
g = Conv1D(16, 5, activation=conv_act)(global_input)
g = Conv1D(16, 5, activation=conv_act)(g)
g = MaxPool1D(5, 2)(g)
g = Conv1D(32, 5, activation=conv_act)(g)
g = Conv1D(32, 5, activation=conv_act)(g)
g = MaxPool1D(5, 2)(g)
g = Conv1D(64, 5, activation=conv_act)(g)
g = Conv1D(64, 5, activation=conv_act)(g)
g = MaxPool1D(5, 2)(g)
g = Conv1D(128, 5, activation=conv_act)(g)
g = Conv1D(128, 5, activation=conv_act)(g)
g = MaxPool1D(5, 2)(g)
g = Conv1D(256, 5, activation=conv_act)(g)
g = Conv1D(256, 5, activation=conv_act)(g)
g = MaxPool1D(5, 2)(g)
g = Flatten()(g)
g = Model(inputs=global_input, outputs=g)

# combine branches
combined = concatenate([l.output, g.output])

# connect layers
z = Dense(256, activation=dense_act)(combined)
z = Dense(256, activation=dense_act)(z)
z = Dense(256, activation=dense_act)(z)
z = Dense(256, activation=dense_act)(z)
z = Dense(2, activation=softmax_act)(z)

# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[l.input, g.input], outputs=z)

model.summary()

opt = Adam(learning_rate=0.01)

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit([l_train, g_train], s_train, epochs=2, batch_size=32)

import matplotlib.pyplot as plt

weights = model.get_weights()
#print(weights)
plt.hist(np.array(weights).flatten(), 10)
plt.show()