
import numpy as np
import pdb
import coremltools

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Permute, Reshape, InputLayer, Flatten


def _keras_transpose(x, is_sequence=False):
    if len(x.shape) == 4:
        # Keras input shape = [Batch, Height, Width, Channels]
        x = np.transpose(x, [0,3,1,2])
        return np.expand_dims(x, axis=0)
    elif len(x.shape) == 3:
        # Keras input shape = [Batch, (Sequence) Length, Channels]
        return np.transpose(x, [1,0,2])
    elif len(x.shape) == 2:
        if is_sequence:  # (N,S) --> (S,N,1,)
            return x.reshape(x.shape[::-1] + (1,))
        else:  # (N,C) --> (N,C,1,1)
            return x.reshape((1, ) + x.shape) # Dense
    elif len(x.shape) == 1:
        if is_sequence: # (S) --> (S,N,1,1,1)
            return x.reshape((x.shape[0], 1, 1))
        else:
            return x
    else:
        return x

def _generate_data(input_shape, mode = 'random'):
    """
    Generate some random data according to a shape.
    """
    if mode == 'zeros':
        X = np.zeros(input_shape)
    elif mode == 'ones':
        X = np.ones(input_shape)
    elif mode == 'linear':
        X = np.array(range(np.product(input_shape))).reshape(input_shape)
    elif mode == 'random':
        X = np.random.rand(*input_shape)
    elif mode == 'random_zero_mean':
        X = np.random.rand(*input_shape)-0.5
    return X

def _get_coreml_model(model, input_names, output_names):
    """
    Get the coreml model from the Keras model.
    """
    # Convert the model
    from coremltools.converters import keras as keras_converter
    model = keras_converter.convert(model, input_names, output_names)
    return model

"""
Unit test function for testing the Keras converter.
"""
def _test_keras_model(model, mode = 'random', delta = 1e-2,
        transpose_keras_result = True):

    # transpose_keras_result: if true, compare the transposed Keras result
    # one_dim_seq_flags: a list of same length as the number of inputs in
    # the model; if None, treat all 1D input (if any) as non-sequence
    # if one_dim_seq_flags[i] is True, it means the ith input, with shape
    # (X,) is in fact a sequence of length X.


    # Generate data
    nb_inputs = len(model.inputs)

    input_shape = [1 if a is None else a for a in model.input_shape]
    input_names = ['data']
    input_data = _generate_data(input_shape, mode)
    coreml_input = {'data': _keras_transpose(input_data).astype('f').copy()}

    # Compile coreml model
    output_names = ['output'+str(i) for i in xrange(len(model.outputs))]
    coreml_model = _get_coreml_model(model, input_names, output_names)

    # Assuming coreml model output names are in the same order as Keras
    # Output list, put predictions into a list, sorted by output name
    coreml_preds = coreml_model.predict(coreml_input)
    c_preds = [coreml_preds[name] for name in output_names]

    # Run Keras predictions
    keras_preds = model.predict(input_data)
    k_preds = keras_preds if type(keras_preds) is list else [keras_preds]

    # Compare each output blob
    pdb.set_trace()
    print(np.array(k_preds) - c_preds)


# cdata.transpose(0,3,1,2).reshape(1,-1,50,50).transpose(0,3,2,1)[0,:,:,0]

if __name__ == '__main__':

    K.set_image_dim_ordering('tf')
    tf_dim_model = Sequential()

    # in case of accepting B x C x H*W x 1  convert to B x H x W x C
    # data = np.arange(160).reshape(10,-1, 1)
    # data.transpose(2,0,1).reshape(-1,4,4).transpose(1,2,0)[:,:,0]
    # #
    # tf_dim_model.add(InputLayer(input_shape=(20, 50*50, 1)))
    # tf_dim_model.add(Permute((3,1,2)))
    # tf_dim_model.add(Reshape((-1, 50, 50)))
    # tf_dim_model.add(Permute((2,3,1)))

    tf_dim_model.add(InputLayer(input_shape=(50, 50, 20)))

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
    tf_dim_model.add(Dense(8, bias=True, activation='softmax')) # 500 -> 8

    tf_dim_model.load_weights('/Users/a_shika/Desktop/Python_Script/DLhacks/AIL2/AIL_team5/kayama/Keras-Classification-Models/tf-kernels-channels-last-dim-ordering/my_weights_theano0.h5')

    # for i in range(13):
    #     tf_dim_model.pop()
    pdb.set_trace()

    _test_keras_model(tf_dim_model)
