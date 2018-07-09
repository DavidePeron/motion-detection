import numpy as np
from numpy.random import seed

from keras import layers
from keras.layers import Input, ZeroPadding2D, Conv2D, ZeroPadding1D, Conv1D, BatchNormalization, Activation, Flatten, Dense
from keras.layers import AveragePooling2D, MaxPooling1D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import regularizers
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from utility import *


import keras.backend as K
K.set_image_data_format('channels_last')

#X = MaxPooling2D((2,1), name = 'max_pool0')(X)

#MODEL FUNCTION CREATED, it requires an input placeholder
def ActivityRecognizer(input_shape):

	#Input method returns a tensor with a shape of input_vector
	X_input = Input(input_shape)

	X = ZeroPadding1D(2)(X_input)

	X = Conv1D(64, 5, strides = 1)(X)
	X = BatchNormalization(axis = 1)(X)
	X = Activation('relu')(X)
	X = Dropout(0.15)(X)

	X = Conv1D(64, 5, strides = 1)(X)
	X = BatchNormalization(axis = 1)(X)
	X = Activation('relu')(X)
	X = Dropout(0.15)(X)
	#X = MaxPooling1D(2, strides = 2)(X)


	X = Conv1D(32, 3, strides = 1)(X)
	X = BatchNormalization(axis = 1)(X)
	X = Activation('relu')(X)
	X = Dropout(0.15)(X)

	X = Conv1D(32, 3, strides = 1)(X)
	X = BatchNormalization(axis = 1)(X)
	X = Activation('relu')(X)

	X = Dropout(0.25)(X)

	#convert into a vector
	X = Flatten()(X)

	#dense=fully connected layer
	#X = Dropout(0.5)(X)
	X = Dense(256, activation = 'relu')(X)

	X = Dropout(0.5)(X)
	X = Dense(128, activation = 'relu')(X)

	X = Dropout(0.5)(X)
	X = Dense(11, activation = 'softmax')(X)

	#this creates the Keras model instance, this instance is gonna be used to train/test the model
	model = Model(inputs = X_input, outputs = X)
	return model

# Returns only X_train and Y_train
X_train, Y_train, X_test, Y_test = load_dataset()

print('X_train shape = ' + str(X_train.shape))
print('Y_train shape = ' + str(Y_train.shape))
print('X_test shape = ' + str(X_test.shape))
print('Y_test shape = ' + str(Y_test.shape))

# Channel(=1) added at the end of each tuple
X_train, Y_train, X_test, Y_test = resize_input(X_train, Y_train, X_test, Y_test)

print('X_train shape = ' + str(X_train.shape))
print('Y_train shape = ' + str(Y_train.shape))
print('X_test shape = ' + str(X_test.shape))
print('Y_test shape = ' + str(Y_test.shape))

# CREATE THE MODEL
activity_recognizer = ActivityRecognizer(X_train.shape[1:])

activity_recognizer.summary()


# COMPILE THE MODEL
activity_recognizer.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#activity_recognizer = load_model('trial7.h5')

# TRAIN THE MODEL
activity_recognizer.fit(x = X_train, y = Y_train, validation_split=0.2, epochs = 10, batch_size = 128) #verbose = 0,

# TEST THE MODEL
loss, acc = activity_recognizer.evaluate(x = X_test, y = Y_test)
print ("Loss = " + str(loss))
print ("Test Accuracy = " + str(acc))



# SAVE THE MODEL
#crates an HDF5 file 'activity_recognizer.h5'
activity_recognizer.save('activity_recognizer.h5')

'''
# LOAD THE MODEL
#if I want to use the activity_recognizer.h5 model to test new patterns
model = load_model('activity_recognizer.h5')
'''
