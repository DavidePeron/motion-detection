import numpy as np
from keras import layers
from keras.layers import Input, ZeroPadding2D, Conv2D, ZeroPadding1D, Conv1D, BatchNormalization, Activation, Flatten, Dense
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
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

	#ZeroPadding pads the input borders with 0
	X = ZeroPadding1D(1)(X_input)

	#those 3 functions compose a single layer
	#32 is the number of output filters in the convolution
	#(7,7) is the kernel size, is the 2D convolution window
	#strides represents how the filter strides along x and y axes
	X = Conv1D(32, 3, strides = 1, name = 'conv0')(X)
	X = BatchNormalization(axis = 1, name = 'bn0')(X)
	X = Activation('relu')(X)


	X = Conv1D(20, 3, strides = 1, name = 'conv1')(X)
	X = BatchNormalization(axis = 1, name = 'bn1')(X)
	X = Activation('relu')(X)


	X = Conv1D(12, 3, strides = 1, name = 'conv2')(X)
	X = BatchNormalization(axis = 1, name = 'bn2')(X)
	X = Activation('relu')(X)

	X = Dropout(0.25, name = 'drop0')(X)

	#convert into a vector
	X = Flatten()(X)

	#dense=fully connected layer
	X = Dense(100, activation = 'sigmoid', name = 'fc0')(X)

	X = Dropout(0.25, name = 'drop1')(X)

	X = Dense(12, activation = 'sigmoid', name = 'fc1')(X)

	#this creates the Keras model instance, this instance is gonna be used to train/test the model
	model = Model(inputs = X_input, outputs = X, name = 'act_recognizer')
	return model

#returns only X_train and Y_train
X_train, Y_train, X_test, Y_test = load_dataset()

print('X_train shape = ' + str(X_train.shape))
print('Y_train shape = ' + str(Y_train.shape))
print('X_test shape = ' + str(X_test.shape))
print('Y_test shape = ' + str(Y_test.shape))

#channel(=1) added at the end of each tuple
X_train, Y_train, X_test, Y_test = resize_input(X_train, Y_train, X_test, Y_test)

print('X_train shape = ' + str(X_train.shape))
print('Y_train shape = ' + str(Y_train.shape))
print('X_test shape = ' + str(X_test.shape))
print('Y_test shape = ' + str(Y_test.shape))
#CREATE THE MODEL
activity_recognizer = ActivityRecognizer(X_train.shape[1:])

activity_recognizer.summary()


#COMPILE THE MODEL
activity_recognizer.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


#TRAIN THE MODEL
activity_recognizer.fit(x = X_train, y = Y_train, shuffle='batch', epochs = 5, batch_size = 128) #verbose = 0,
'''
#TEST THE MODEL
preds = activity_recognizer.evaluate(x = X_test, y = Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
'''