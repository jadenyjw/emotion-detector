from __future__ import print_function
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.misc import imsave
import scipy
matplotlib.use("TkAgg")

h5f = h5py.File('dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']

def format_image(image):
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

  image = np.reshape(image, [48, 48, 3])
  return image
'''
count = 0
from matplotlib import pyplot as plt
P = np.empty([48, 48, 3], 'float32')
for i in range (len(X)):

    max = 0.
    for t in range(0, 6):
        if Y[i][t] > max:
            max = Y[i][t]
            maxIndex = t
    if maxIndex == 3:
        count += 1
        P = P + X[i]

P = P/count
plt.imshow(P, interpolation='nearest')
plt.show()
face = scipy.misc.imread('face.png')
print(face)
imsave('original.png', format_image(face))
imsave('after.png', format_image(face) - P)
'''
mean = 0.
std = 0.

face = format_image(scipy.misc.imread('face.png'))
face = face - np.mean(face, axis=0)
face /= np.std(face, axis=0)

print(face)

imsave('original.png', face)
#imsave('lol.png', new)
plt.imshow(face)
plt.show()
