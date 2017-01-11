import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import DataPreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.layers.normalization import local_response_normalization
import h5py
import matplotlib
import numpy as np


h5f = h5py.File('accuracy.h5', 'r')
X = h5f['X']
Y = h5f['Y']

img_prep = DataPreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 48, 48, 3])
network = conv_2d(network, nb_filter= 64, filter_size= [5, 5], activation='relu')
network = local_response_normalization(network)
network = max_pool_2d(network, kernel_size=[3, 3], strides=2)
network = conv_2d(network, nb_filter= 64, filter_size= [5, 5],  activation='relu')
network = max_pool_2d(network, kernel_size=[3, 3], strides=2)
network = conv_2d(network, nb_filter= 128, filter_size= [4, 4], activation='relu')
network = dropout(network, 0.3)
network = fully_connected(network, 3072, activation='relu')
network = fully_connected(network, 7, activation ='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy')
model = tflearn.DNN(network, tensorboard_verbose=0)
model.load("/home/jaden/tensorflow/emotion/emotion.tfl")

'''
print(len(X))
print(X[0])
print(Y[0])
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")
plt.imshow(X[0], interpolation='nearest')
plt.show()
'''



confusion = np.zeros([7,7], 'int')

count = 0
for i in range(0,5):
    print(Y[i])
    prediction = model.predict(X[i].reshape(-1,48,48,3))
    #print(prediction)
    max = 0.

    for b in range(len(prediction[0])):
        if prediction[0][b] > max:
            max = prediction[0][b]
            maxIndex = b

    for c in range(len(Y[i])):
        if Y[i][c] == 1:
            maxYIndex = c

    if maxIndex == maxYIndex:
        count += 1

    print(maxIndex)
    confusion[maxYIndex][maxIndex] += 1

print(confusion)
print(str(count) + '/' + str(len(X)))
