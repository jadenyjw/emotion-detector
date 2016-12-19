from __future__ import print_function
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use("TkAgg")

h5f = h5py.File('dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']

from matplotlib import pyplot as plt

for i in range (10):
    plt.imshow(X[i], interpolation='nearest')
    plt.show()
    print(Y[i])
