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

training_file = 'train_data.txt'
validation_file = 'cval_data.txt'


#build_hdf5_image_dataset(training_file, image_shape=[48, 48, 3], mode='file', output_path='dataset.h5', categorical_labels=True)

h5f = h5py.File('dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']

#build_hdf5_image_dataset(validation_file, image_shape=[48, 48, 3], mode='file', output_path='validation.h5', categorical_labels=True)

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
#network = local_response_normalization(network)
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
model.fit(X, Y, n_epoch=60, shuffle=True, validation_set=(X_val, Y_val),
          show_metric=True, batch_size=50,
          snapshot_epoch=True,
          run_id='emotionv2')

# Save model when training is complete to a file
model.save("emotion.tfl")
