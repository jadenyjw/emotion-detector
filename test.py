from scipy import ndimage
from numpy import genfromtxt
import numpy as np
import scipy.misc

import string
import random

import os

f = open('train_data.txt', 'a')
c = open('cval_data.txt', 'a')
to = open('test_data.txt' , 'a')


csv = genfromtxt('fer2013.csv', delimiter=",", dtype=None)

for i in range(1, len(csv)):
    t = str(csv[i,1]).replace("b", "").replace("'", "").split(" ")
    data = np.reshape(t, (48, 48)).astype(np.float)

    if str(csv[i,2]).replace("b", "").replace("'", "").split(" ") == ['Training']:
        thing = "training"
        d = f
    elif str(csv[i,2]).replace("b", "").replace("'", "").split(" ") == ['PulicTest']:
        thing = "cval"
        d = c
    elif str(csv[i,2]).replace("b", "").replace("'", "").split(" ") == ['PrivateTest']:
        thing = "test"
        d = to
    else:
        print(str(csv[i,2]).replace("b", "").replace("'", "").split(" "))

    scipy.misc.imsave("data_new/" + thing + "/" + str(i) + ".png", data)

    if str(csv[i,0]).replace("b", "").replace("'", "").split(" ") == ['0']:
        d.write("data_new/" + thing + "/" + str(i) + ".png" + ' 0\n')
    elif str(csv[i,0]).replace("b", "").replace("'", "").split(" ") == ['1']:
        d.write("data_new/" + thing + "/" + str(i) + ".png" + ' 1\n')
    elif str(csv[i,0]).replace("b", "").replace("'", "").split(" ") == ['2']:
        d.write("data_new/" + thing + "/" + str(i) + ".png" + ' 2\n')
    elif str(csv[i,0]).replace("b", "").replace("'", "").split(" ") == ['3']:
        d.write("data_new/" + thing + "/" + str(i) + ".png" + ' 3\n')
    elif str(csv[i,0]).replace("b", "").replace("'", "").split(" ")== ['4']:
        d.write("data_new/" + thing + "/" + str(i) + ".png" + ' 4\n')
    elif str(csv[i,0]).replace("b", "").replace("'", "").split(" ")== ['5']:
        d.write("data_new/" + thing + "/" + str(i) + ".png" + ' 5\n')
    elif str(csv[i,0]).replace("b", "").replace("'", "").split(" ") == ['6']:
        d.write("data_new/" + thing + "/" + str(i) + ".png" + ' 6\n')
