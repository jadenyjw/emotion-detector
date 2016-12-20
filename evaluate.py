import cv2
import sys
import os
import numpy as np

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
import scipy
from scipy import ndimage
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use("TkAgg")

parser = argparse.ArgumentParser(description='Find the emotion of the image.')
parser.add_argument('image', type=str, help='The image image file to check')
args = parser.parse_args()

def format_image(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(
      image,
      scaleFactor = 1.3,
      minNeighbors = 5
  )
  # None is we don't found an image
  if not len(faces) > 0:
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
  # Chop image to face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
  image = scipy.misc.imresize(image, [48, 48])
  from matplotlib import pyplot as plt
  matplotlib.use("TkAgg")

  plt.imshow(image, interpolation='nearest')
  plt.show()


  image = np.reshape(image, [-1, 48, 48, 3])
  return image

h5f = h5py.File('dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']

h5f = h5py.File('validation.h5', 'r')
X_val = h5f['X']
Y_val = h5f['Y']

X, Y = shuffle(X, Y)

img_prep = DataPreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 48, 48, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, nb_filter= 64, filter_size= [5, 5], activation='relu')
network = local_response_normalization(network)
network = max_pool_2d(network, kernel_size=[3, 3], strides=2)
network = conv_2d(network, nb_filter= 64, filter_size= [5, 5],  activation='relu')
network = max_pool_2d(network, kernel_size=[3, 3], strides=2)
network = conv_2d(network, nb_filter= 128, filter_size= [4, 4], activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 3072, activation='relu')
#network = fully_connected(network, 3072, activation='relu')
network = fully_connected(network, 7, activation ='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy')
model = tflearn.DNN(network, tensorboard_verbose=0)

model.load("emotion.tfl")

img = scipy.misc.imread(args.image)

prediction = model.predict(format_image(img).astype('float32'))
print(prediction)
max = 0.
for i in range(len(prediction[0])):
    if prediction[0][i] > max:
        max = prediction[0][i]
        maxIndex = i
print(maxIndex)
