
import vlc
import sys
import tkinter as Tk
from tkinter import ttk

import string
import random
# import standard libraries
import os
import pathlib
from threading import Thread, Event
import time
import platform
import argparse

import cv2
import sys
import os
import numpy as np

import time
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


parser = argparse.ArgumentParser(description='Wireless controller of CarNet.')
parser.add_argument('camera', type=str, help='The IP address of the remote camera.')
#parser.add_argument('car', type=str, help='The IP address of the car.')
args = parser.parse_args()

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

  image = np.reshape(image, [-1, 48, 48, 3])
  return image



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

model.load("/home/jaden/tensorflow/emotion/emotion.tfl")


class Player(Tk.Frame):
    """The main window has to deal with events.
    """
    def __init__(self, parent, title=None):
        Tk.Frame.__init__(self, parent)

        self.parent = parent

        if title == None:
            title = "Carnet"
        self.parent.title(title)

        self.player = None
        self.videopanel = ttk.Frame(self.parent)
        self.canvas = Tk.Canvas(self.videopanel).pack(fill=Tk.BOTH,expand=1)
        self.videopanel.pack(fill=Tk.BOTH,expand=1)

        # VLC player controls
        self.Instance = vlc.Instance()
        self.player = self.Instance.media_player_new()

        # below is a test, now use the File->Open file menu
        media = self.Instance.media_new('rtsp://' + args.camera + ':5554/playlist.m3u')
        self.player.set_media(media)
        self.player.play() # hit the player button
        self.player.video_set_deinterlace(str.encode('yadif'))


        self.parent.update()
        self.player.set_xwindow(self.GetHandle())
        time.sleep(4)
        while 1:
            self.player.video_take_snapshot(0, "picture.png", 0, 0)
            time.sleep(1)
            img = scipy.misc.imread("picture.png")
            print(format_image(img))
            if format_image(img) != None:
                prediction = model.predict(format_image(img).astype('float32'))
                print(prediction)
                max = 0.
                for i in range(len(prediction[0])):
                    if prediction[0][i] > max:
                        max = prediction[0][i]
                        maxIndex = i
                print(maxIndex)


    def GetHandle(self):
        return self.videopanel.winfo_id()


def Tk_get_root():
    if not hasattr(Tk_get_root, "root"): #(1)
        Tk_get_root.root= Tk.Tk()  #initialization call is inside the function
    return Tk_get_root.root

def _quit():
    print("_quit: bye")
    root = Tk_get_root()
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    os._exit(1)

if __name__ == "__main__":
    # Create a Tk.App(), which handles the windowing system event loop
    root = Tk_get_root()
    root.protocol("WM_DELETE_WINDOW", _quit)

    player = Player(root, title="CarNet")

    #buttons = Tk.Frame(root)
    # show the player window centred and run the application
    root.mainloop()
