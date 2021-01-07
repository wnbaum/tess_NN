import numpy as np

import matplotlib.pyplot as plt

file = "298554256"

print(np.load("C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/processed/" + file + "_g.npy"))
print(np.load("C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/processed/" + file + "_l.npy"))
print(np.load("C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/processed/" + file + "_y.npy"))
print(np.load("C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/processed/" + file + "_s.npy"))

plt.plot(np.load("C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/processed/" + file + "_g.npy"))
plt.show()
plt.plot(np.load("C:/Users/wnbau/Documents/Programming Projects/NeuralNetworks/Astronomy/processed/" + file + "_l.npy"))
plt.show()