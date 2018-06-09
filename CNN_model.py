import numby as np
from keras import layers
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Flatten
#from keras.layers import
from keras.model import Model
from keras import

#from {FILE_CON_FUNZIONI_CARICAMENTO_DATSET} import*


#load_dataset() function needs to be created
 X_train, Y_trian, X_test, Y_test = load_dataset()


print('X_train shape = ' + str(X_train.shape))
print('Y_train shape = ' + str(Y_train.shape))
print('X_test shape = ' + str(X_test.shape))
print('Y_test shape = ' + str(Y_test.shape))


#placeholder will have 5***x9 shape
#model function created, it requires an input placeholder
def model(input_shape):
 	#Input method returns a tensor with a shape of input_vector
 	X_input = Input(input_shape)

 	#ZeroPadding pads the input borders with 0
 	#(3,3) = (symmetric_height_pad, symmetric_width_pad)
 	X = ZeroPadding2D((3,3))(X_input)

 	#those 3 functions compose a single layer
 	#32 is the number of output filters in the convolution
 	#(7,7) is the kernel size, is the 2D convolution window
 	#strides represents how the filter strides along x and y axes
 	X = Conv2D(32, (3,3), strides = (1,1), name = 'conv0')(X)
 	X = MaxPooling2D((2,2), name = 'max_pool')(X)

	X = BatchNormalization(axis = 3, name = 'bn0')(X)
 	X = MaxPooling2D((2,2), name = 'max_pool')(X)

	X = Activation('relu')(X)
	X = MaxPooling2D((2,2), name = 'max_pool')(X)

 	#convert into a vector
 	X = Flatten()(X)

 	#dense=fully connected layer
 	X = Dense(1, activation = 'sigmoid', name = 'fc')(X)

 	#thia cretes the Keras model instance, this instance is gonna be used to train/test the model
 	model = Model(inputs = X_input, outputs = X, name = 'MyModel')

 	return model



 	#CREATE THE MODEL
 	happyModel = HappyModel((64,64,3))
 	
 	#COMPILE THE MODEL
 	happyModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
 	
 	#TRAIN THE MODEL
 	happyModel.fit(x = X_train, y = Y_train, epochs = 40, batch_size = 32)

 	#TEST THE MODEL
	preds = happyModel.evaluate(x = X_test, y = Y_test)

	print()
	print ("Loss = " + str(preds[0]))
	print ("Test Accuracy = " + str(preds[1]))