import os
import numpy as np

from keras import backend as K
# from keras.utils.layer_utils import convert_all_kernels_in_model # keras2.0
from keras.utils.np_utils import convert_kernel # keras1.2

''' IMPORT YOUR SCRIPT FILE HERE TO CREATE YOUR MODEL LATER '''


''' BACKEND must be TENSORFLOW

This is a script to convert Theano models (Theano Backend, TH dim ordering)
to the other possible backend / dim ordering combinations.

Given weights and model for TH-kernels-TH-dim-ordering, produces a folder with
- TH-kernels-TF-dim-ordering
- TF-kernels-TH-dim-ordering
- TF-kernels-TF-dim-ordering

Needs 3 important inputs:

1) Theano model (model with TH dim ordering)
2) Tensorflow model (model with TF dim ordering)
3) Weight file for Theano model (theano-kernels-th-dim-ordering)

Supports : Multiple weights for same model (auto converts different weights for same model)

Usage:
1) Place script in the same directory as the weight file directory. If you want to place somewhere
   else, then you must provide absolute path to the weight files below instead of relative paths.

2) Edit the script to create your model :
    a) Import your model building script above (in the imports section)
    b) Set `th_dim_model` = ... (create your th dim model here and set it to th_dim_model)
    c) Set `tf_dim_model` = ... (create your tf dim model here and set it to tf_dim_model)
    d) Add the path to the weight files in `model_weights`.
       Note : The weight files must be for the Theano model (theano kernels, th dim ordering)

3) Run the script.

4) Use the weight files in the created folders : ["tf-kernels-tf-dim/", "tf-kernels-th-dim/", "th-kernels-tf-dim/"]
'''


from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

K.set_image_dim_ordering('th')
# th_dim_model = None # Create your theano model here with TH dim ordering
th_dim_model = Sequential() # Create your tensorlow model with TF dimordering here
th_dim_model.add(InputLayer(input_shape=(20, 50,50)))
th_dim_model.add(Convolution2D(40, 5, 5, bias=True))
th_dim_model.add(MaxPooling2D(pool_size=(2,2)))
th_dim_model.add(Activation('relu'))
th_dim_model.add(Convolution2D(60, 3, 3, bias=True))
th_dim_model.add(MaxPooling2D(pool_size=(2,2)))
th_dim_model.add(Activation('relu'))
th_dim_model.add(Convolution2D(90, 3, 3, bias=True))
th_dim_model.add(MaxPooling2D(pool_size=(2,2)))
th_dim_model.add(Activation('relu'))
th_dim_model.add(Flatten())
th_dim_model.add(Dense(500, bias=True)) # 1440 -> 500
th_dim_model.add(Activation('tanh'))
th_dim_model.add(Dense(8, bias=True)) # 500 -> 8

K.set_image_dim_ordering('tf')
# tf_dim_model = None # Create your tensorflow model with TF dimordering here
tf_dim_model = Sequential() # Create your tensorflow model with TF dimordering here
tf_dim_model.add(InputLayer(input_shape=(50,50, 20)))
tf_dim_model.add(Convolution2D(40, 5, 5, bias=True))
tf_dim_model.add(MaxPooling2D(pool_size=(2,2)))
tf_dim_model.add(Activation('relu'))
tf_dim_model.add(Convolution2D(60, 3, 3, bias=True))
tf_dim_model.add(MaxPooling2D(pool_size=(2,2)))
tf_dim_model.add(Activation('relu'))
tf_dim_model.add(Convolution2D(90, 3, 3, bias=True))
tf_dim_model.add(MaxPooling2D(pool_size=(2,2)))
tf_dim_model.add(Activation('relu'))
tf_dim_model.add(Flatten())
tf_dim_model.add(Dense(500, bias=True)) # 1440 -> 500
tf_dim_model.add(Activation('tanh'))
tf_dim_model.add(Dense(8, bias=True)) # 500 -> 8

model_weights = ['my_weights_theano0.h5'] # Add names of theano model weight file paths here.
                     # These weights are assumed to be for  theano backend
                     # (th kernels) with th dim ordering!

"""

No need to edit anything below this. Simply run the script now after
editing the above 3 inputs.

"""


def shuffle_rows(original_w, nb_last_conv, nb_rows_dense):
    ''' Note :
    This algorithm to shuffle dense layer rows was provided by Kent Sommers (@kentsommer)
    in a gist : https://gist.github.com/kentsommer/e872f65926f1a607b94c2b464a63d0d3
    '''
    converted_w = np.zeros(original_w.shape)
    count = 0
    for index in range(original_w.shape[0]):
        if (index % nb_last_conv) == 0 and index != 0:
            count += 1
        new_index = ((index % nb_last_conv) * nb_rows_dense) + count
        print("index from " + str(index) + " -> " + str(new_index))
        converted_w[index] = original_w[new_index]

    return converted_w

# convert tf kernel -> th kernel Keras1.2 (converting dimension is different)
def convert_all_kernels_in_model(model, dim_ordering):
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
            original_w = layer.get_weights()[0]
            original_b = layer.get_weights()[1]
            converted_w = convert_kernel(original_w, dim_ordering)
            converted_parameter = [converted_w, original_b]
            layer.set_weights(converted_parameter)


first_dense = True
nb_last_conv = 0

for dirpath in ["tf-kernels-channels-last-dim-ordering/", "tf-kernels-channels-first-dim-ordering/", "th-kernels-channels-last-dim-ordering/"]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Converts (theano kernels, th dim ordering) to (tensorflow kernels, th dim ordering)
K.set_image_dim_ordering('tf')
for weight_fn in model_weights:
    th_dim_model.load_weights(weight_fn)
    # Converts (theano kernels, th dim ordering) to (tensorflow kernels, th dim ordering)
    convert_all_kernels_in_model(th_dim_model, dim_ordering='th')

    th_dim_model.save_weights("tf-kernels-channels-first-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done tf-kernels-channels-first-dim-ordering %s" % weight_fn)


# Converts (theano kernels, th dim ordering) to (tensorflow kernels, tf dim ordering)
K.set_image_dim_ordering('th')
for weight_fn in model_weights:
    th_dim_model.load_weights(weight_fn) # th-kernels-th-dim

    # Converts (theano kernels, th dim ordering) to (tensorflow kernels, th dim ordering)
    convert_all_kernels_in_model(th_dim_model, dim_ordering='th') # tf-kernels-th-dim

    count_dense = 0
    for layer in th_dim_model.layers:
        if layer.__class__.__name__ == "Dense":
            count_dense += 1

    if count_dense == 1:
        first_dense = False # If there is only 1 dense, no need to perform row shuffle in Dense layer

    print("Nb layers : ", len(th_dim_model.layers))

    for index, th_layer in enumerate(th_dim_model.layers):
        if th_layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
            weights = th_layer.get_weights() # tf-kernels-th-dim
            weights[0] = weights[0].transpose((2, 3, 1, 0))
            tf_dim_model.layers[index].set_weights(weights) # tf-kernels-tf-dim

            nb_last_conv = th_layer.nb_filter # preserve last number of convolutions to use with dense layers
            print("Converted layer %d : %s" % (index + 1, th_layer.name))
        else:
            if th_layer.__class__.__name__ == "Dense" and first_dense:
                weights = th_layer.get_weights()
                nb_rows_dense_layer = weights[0].shape[0] // nb_last_conv

                print("Magic Number 1 : ", nb_last_conv)
                print("Magic nunber 2 : ", nb_rows_dense_layer)

                weights[0] = shuffle_rows(weights[0], nb_last_conv, nb_rows_dense_layer)
                tf_dim_model.layers[index].set_weights(weights)

                first_dense = False
                print("Shuffled Dense Weights layer and saved %d : %s" % (index + 1, th_layer.name))
            else:
                tf_dim_model.layers[index].set_weights(th_layer.get_weights())
                print("Saved layer %d : %s" % (index + 1, th_layer.name))


    tf_dim_model.save_weights("tf-kernels-channels-last-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done tf-kernels-channels-last-dim-ordering %s" % weight_fn)


# Converts (theano kernels, th dim ordering) to (theano kernels, tf dim ordering)
for weight_fn in model_weights:
    th_dim_model.load_weights(weight_fn)

    for index, th_layer in enumerate(th_dim_model.layers):
        if th_layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
            weights = th_layer.get_weights()
            weights[0] = weights[0].transpose((2, 3, 1, 0))
            tf_dim_model.layers[index].set_weights(weights)
        else:
            tf_dim_model.layers[index].set_weights(th_layer.get_weights())

        print("Changed dim %d : %s" % (index + 1, th_layer.name))

    tf_dim_model.save_weights("th-kernels-channels-last-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done th-kernels-channels-last-dim-ordering %s" % weight_fn)
