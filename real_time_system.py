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

activity_recognizer = load_model('trial8.h5')
