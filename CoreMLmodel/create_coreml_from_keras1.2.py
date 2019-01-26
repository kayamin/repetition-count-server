#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import pdb
import numpy as np

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import InputLayer, Reshape, Permute, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# # Keras1.2.2 case
K.set_image_dim_ordering('tf')
tf_dim_model = Sequential() # Create your tensorflow model with TF dimordering here
# tf_dim_model.add(InputLayer(input_shape=(50,50, 1)))

# # in case of accepting B x H x W*C x 1 input
# tf_dim_model.add(InputLayer(input_shape=(50, 50*20, 1)))
# tf_dim_model.add(Permute((3,2,1)))
# tf_dim_model.add(Reshape((-1, 50, 50)))
# tf_dim_model.add(Permute((3,2,1)))


# in case of accepting B x C x H*W x 1  convert to B x H x W x C
# data = np.arange(160).reshape(10,-1, 1)
# data.transpose(2,0,1).reshape(-1,4,4).transpose(1,2,0)[:,:,0]

# tf_dim_model.add(InputLayer(input_shape=(20, 50*50, 1)))
# tf_dim_model.add(Permute((3,1,2)))
# tf_dim_model.add(Reshape((-1, 50, 50)))
# tf_dim_model.add(Permute((2,3,1)))
pdb.set_trace()
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

# Convert keras model to coreml model
pdb.set_trace()
import coremltools

# # in case of using image as input
# output_labels = ['3', '4', '5', '6', '7', '8', '9', '10']
# coreml_model = coremltools.converters.keras.convert(tf_dim_model,
#                                                     input_names = 'image',
#                                                     image_input_names = 'image',
#                                                     output_names = 'output',
#                                                     class_labels = output_labels,
#                                                     )
# coreml_model.author = 'Atsushi Kayama'
#
# coreml_model.short_description = 'Model to classify cycle length of motion within 20frames'
# coreml_model.input_description['image'] = 'Grayscale image of H x W*C x 1 -> 50 x (50*20) x 1'
# coreml_model.output_description['output'] = 'Predicted cyclelength label and softmaxed probability'


# in case of using MLMultiarray as input
output_labels = ['3', '4', '5', '6', '7', '8', '9', '10']
coreml_model = coremltools.converters.keras.convert(tf_dim_model,
                                                    input_names = 'data',
                                                    output_names = 'output',
                                                    class_labels = output_labels,
                                                    )
coreml_model.author = 'Atsushi Kayama'

coreml_model.short_description = 'Model to classify cycle length of motion within 20frames. Created at 2019/1/21'
coreml_model.input_description['data'] = 'Grayscale image Tensor of C x H x W -> 20 x 50 x 50'
coreml_model.output_description['output'] = 'Predicted cyclelength label and softmaxed probability'

coreml_model.save('RepetitionCounting.mlmodel')

coreml_input = {}
coreml_input['data'] = np.ones([1, 1, 20, 50, 50])
print(coreml_model.predict(coreml_input))

pdb.set_trace()

# 変換した coreml モデルのテスト





# coreml では 入力が C x H x W の形式で与えられることになっている

data = np.arange(40).reshape(1,1,2,2*10)

data.transpose(0,1,3,2).reshape(1,-1,2,2).transpose(0,1,3,2)[0,0,:,:]

# 素直な実装
tf_dim_model.add(InputLayer(input_shape=(1, 50, 40*20)))
tf_dim_model.add(Permute((1,3,2)))
tf_dim_model.add(Reshape((-1, 40, 50)))
tf_dim_model.add(Permute((1,3,2)))


tf_dim_model.add(Permute((3,2,1)))
tf_dim_model.add(Reshape((-1, 40, 50)))
tf_dim_model.add(Permute((3,2,1)))


# coreML内で勝手に変わってしまうことを考慮した実装
tf_dim_model.add(InputLayer(input_shape=(50, 40*20, 1))) # 最後の次元を先頭に勝手に持ってくるため
tf_dim_model.add(Permute((3,2,1)))
tf_dim_model.add(Reshape((40, 50, -1)))
tf_dim_model.add(Permute((3,2,1)))



# Keras1.2.2 case
import keras.backend as K
from keras.models import Sequential
from keras.layers import InputLayer, Reshape, Permute, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


K.set_image_dim_ordering('tf')
tf_dim_model = Sequential() # Create your tensorflow model with TF dimordering here
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

tf_dim_model.load_weights('./Keras-Classification-Models/tf-kernels-channels-last-dim-ordering/my_weights_theano0.h5')

# Convert keras model to coreml model
# in case of using MLMultiarray as input
import coremltools
output_labels = ['3', '4', '5', '6', '7', '8', '9', '10']
coreml_model = coremltools.converters.keras.convert(tf_dim_model,
                                                    input_names = 'data',
                                                    output_names = 'output',
                                                    class_labels = output_labels,
                                                    )
coreml_model.author = 'Atsushi Kayama'

coreml_model.short_description = 'Model to classify cycle length of motion within 20frames. Created at 2019/1/21'
coreml_model.input_description['data'] = 'Grayscale image Tensor of C x H x W -> 20 x 50 x 50'
coreml_model.output_description['output'] = 'Predicted cyclelength label and softmaxed probability'

coreml_model.save('RepetitionCounting.mlmodel')