import os
import numpy as np

import tensorflow as tf

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

size = 700

l_test = l_train[size:1515]
g_test = g_train[size:1515]
s_test = s_train[size:1515]
y_test = y_train[size:1515]

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

l_test = np.array(l_test)
g_test = np.array(g_test)
s_test = np.array(s_test) 
y_test = np.array(y_test) 

l_train[np.isnan(l_train)] = 0
g_train[np.isnan(g_train)] = 0
s_train[np.isnan(s_train)] = 0
y_train[np.isnan(y_train)] = 0

l_test[np.isnan(l_test)] = 0
g_test[np.isnan(g_test)] = 0
s_test[np.isnan(s_test)] = 0
y_test[np.isnan(y_test)] = 0

print(l_train[0])

import keras
from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPool1D, Input, concatenate, Flatten
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, SGD

# define two sets of inputs
local_input = Input(shape=(100,)) # 100 bins
global_input = Input(shape=(1000,)) # 1000 bins

conv_act = "relu"
dense_act = "relu"
softmax_act = "relu"

# local branch
l = Dense(1024, activation=softmax_act)(local_input)
l = Dense(512, activation=softmax_act)(l)
l = Dense(256, activation=softmax_act)(l)
l = Dense(128, activation=softmax_act)(l)
l = Dense(64, activation=softmax_act)(l)
l = Dense(32, activation=softmax_act)(l)
l = Dense(16, activation=softmax_act)(l)
l = Dense(8, activation=softmax_act)(l)
l = Dense(4, activation=softmax_act)(l)
l = Dense(2, activation='softmax')(l)
l = Model(inputs=local_input, outputs=l)

l.summary()

opt = Adam(learning_rate=0.01)

l.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = l.fit(l_train, s_train, epochs=1000, batch_size=32)

results = l.evaluate(l_test, s_test, batch_size=32)
print("test loss, test acc:", results)

import matplotlib.pyplot as plt

weights = l.get_weights()
#print(weights)
plt.hist(np.array(weights).flatten(), 10)
plt.show()