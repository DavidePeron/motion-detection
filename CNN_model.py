import numpy as np
from keras import layers
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Flatten, Dense
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from utility import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow



print('inizio')

#X = MaxPooling2D((2,1), name = 'max_pool0')(X)

#MODEL FUNCTION CREATED, it requires an input placeholder
def ActivityRecognizer(input_shape):

	#Input method returns a tensor with a shape of input_vector
	X_input = Input(input_shape)


	#ZeroPadding pads the input borders with 0
	#(n,m) = (symmetric_height_pad, symmetric_width_pad)
	X = ZeroPadding2D((1,0))(X_input)


	#those 3 functions compose a single layer
	#32 is the number of output filters in the convolution
	#(7,7) is the kernel size, is the 2D convolution window
	#strides represents how the filter strides along x and y axes
	X = Conv2D(1, (3,3), strides = (1,1), name = 'conv0')(X)
	X = BatchNormalization(axis = 3, name = 'bn0')(X)
	X = Activation('relu')(X)

	X = ZeroPadding2D((1,0))(X_input)

	X = Conv2D(1, (3,3), strides = (1,1), name = 'conv0')(X)
	X = BatchNormalization(axis = 3, name = 'bn0')(X)
	X = Activation('relu')(X)


	#ZeroPadding pads the input borders with 0
	#(n,m) = (symmetric_height_pad, symmetric_width_pad)
	X = ZeroPadding2D((1,0))(X_input)

	X = Conv2D(1, (3,3), strides = (1,1), name = 'conv0')(X)
	X = BatchNormalization(axis = 3, name = 'bn0')(X)
	X = Activation('relu')(X)

	
	X = ZeroPadding2D((1,0))(X_input)

	X = Conv2D(1, (3,3), strides = (1,1), name = 'conv0')(X)
	X = BatchNormalization(axis = 3, name = 'bn0')(X)
	X = Activation('relu')(X)	

	#convert into a vector
	X = Flatten()(X)

	#dense=fully connected layer
	X = Dense(1, activation = 'sigmoid', name = 'fc')(X)

	#this cretes the Keras model instance, this instance is gonna be used to train/test the model
	model = Model(inputs = X_input, outputs = X, name = 'act_recognizer')
	return model



print('0')

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

print('1')
#CREATE THE MODEL
activity_recognizer = ActivityRecognizer((1000,9,1))

print('2')

activity_recognizer.summary()

print('3')

#COMPILE THE MODEL
activity_recognizer.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])

'''
print('4')
#TRAIN THE MODEL
activity_recognizer.fit(x = X_train, y = Y_train, epochs = 50, batch_size = 32)
print('5')

#TEST THE MODEL
preds = activity_recognizer.evaluate(x = X_test, y = Y_test)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
'''
