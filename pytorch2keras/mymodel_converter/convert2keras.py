#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import pdb
import numpy as np

import keras
import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../pytorch2keras')
from converter import pytorch_to_keras
from RepCountNet import RepetitionCountingNet


# Convert pytorch model to keras model
model = RepetitionCountingNet('./weights.save')

input_np = np.random.uniform(0, 1, (1, 20, 50, 50))
input_var = Variable(torch.FloatTensor(input_np))
output   = model(input_var)
pytorch_output = output.data.numpy()

#
# k_model = pytorch_to_keras((20, 50, 50), output, change_ordering=False)
# k_model.save_weights('my_weights_theano.h5')
# keras_output = k_model.predict(input_np)
#
# pdb.set_trace()

# k_model = pytorch_to_keras((20, 50, 50), output, change_ordering=True)
# k_model.load_weights('my_weights_theano.h5')
#
# from keras import backend as K
# # from keras.utils.conv_utils import convert_kernel # keras2.0
# from keras.utils.np_utils import convert_kernel # keras1.2

import tensorflow as tf
ops = []

# convert tf kernel -> th kernel Keras1.2 (converting dimension is different)
# for layer in k_model.layers:
#     if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
#         # original_w = K.get_value(layer.W)
#         # converted_w = convert_kernel(original_w)
#         # pdb.set_trace()
#         # ops.append(tf.assign(layer.W, converted_w).op)
#         original_w = layer.get_weights()[0]
#         original_b = layer.get_weights()[1]
#         converted_w = convert_kernel(original_w, dim_ordering='th')
#         pdb.set_trace()
#         converted_parameter = [converted_w, original_b]
#         layer.set_weights(converted_parameter)

# convert tf kernel <-> th kernel Keras2.0 (coverting dimension is same so reversible)
# for layer in k_model.layers:
#     if layer.__class__.__name__ in ['Con1D', 'Conv2D', 'Conv3D', 'AtrousConvolution2D']:
#
        # original_w = layer.get_weights()[0]
        # original_b = layer.get_weights()[1]
        # converted_w = convert_kernel(original_w)
        # pdb.set_trace()
        # converted_parameter = [converted_w, original_b]
        # layer.set_weights(converted_parameter)

# convert channel_first -> channel_last
pdb.set_trace()
import keras.backend as K
from keras.models import Sequential
from keras.layers import InputLayer, Reshape, Permute, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


# # Keras 2.0.6 case
# K.set_image_data_format('channels_last')
# tf_dim_model = Sequential() # Create your tensorflow model with TF dimordering here
# tf_dim_model.add(InputLayer(input_shape=(50,50, 20)))
# tf_dim_model.add(Convolution2D(40, 5, use_bias=True))
# tf_dim_model.add(MaxPooling2D(pool_size=(2,2)))
# tf_dim_model.add(Activation('relu'))
# tf_dim_model.add(Convolution2D(60, 3, use_bias=True))
# tf_dim_model.add(MaxPooling2D(pool_size=(2,2)))
# tf_dim_model.add(Activation('relu'))
# tf_dim_model.add(Convolution2D(90, 3, use_bias=True))
# tf_dim_model.add(MaxPooling2D(pool_size=(2,2)))
# tf_dim_model.add(Activation('relu'))
# tf_dim_model.add(Flatten())
# tf_dim_model.add(Dense(500)) # 1440 -> 500
# tf_dim_model.add(Activation('tanh'))
# tf_dim_model.add(Dense(8)) # 500 -> 8
#
# tf_dim_model.save_weights('my_weights_theano.h5')

# # Keras1.2.2 case
K.set_image_dim_ordering('tf')
tf_dim_model = Sequential() # Create your tensorflow model with TF dimordering here
tf_dim_model.add(InputLayer(input_shape=(50,50, 1)))

# # in case of accepting B x H x W*C x 1 input
# tf_dim_model.add(InputLayer(input_shape=(50, 50*20, 1)))
# tf_dim_model.add(Permute((3,2,1)))
# tf_dim_model.add(Reshape((-1, 50, 50)))
# tf_dim_model.add(Permute((3,2,1)))
#
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

if layer.split('_')[1] == '2':↲
 xtmp = x↲
 x[0], x[1], x[2] = xtmp[2], xtmp[0], xtmp[1]
# tf_dim_model.load_weights('/Users/a_shika/Desktop/Python_Script/DLhacks/AIL2/AIL_team5/kayama/Keras-Classification-Models/tf-kernels-channels-last-dim-ordering/my_weights_theano0.h5')

# for index, layer in enumerate(tf_dim_model.layers):
#     if layer.__class__.__name__ in ['Conv1D',
#                                        'Conv2D',
#                                        'Conv3D',
#                                        'AtrousConvolution1D'
#                                        'AtrousConvolution2D',
#                                        'Conv2DTranspose',
#                                        'SeparableConv2D',
#                                        'DepthwiseConv2D',
#                                        ]:
#         weights = layer.get_weights()
#         weights[0] = weights[0].transpose((2, 3, 1, 0))
#         pdb.set_trace()
#         tf_dim_model.layers[index].set_weights(weights) # 次元を入れ替えた重みをそのまま代入することは出来ないので，予めchannel_last形式で定義していたモデルに代入する形で重みを残す
#     else:
#         pdb.set_trace()
#         tf_dim_model.layers[index].set_weights(layer.get_weights())
#
# tf_dim_model.save_weights('my_weights_tf_kernel_channel_last.h5')

# keras_output = k_model.predict(input_np)


# K.get_session().run(ops)
# k_model.save_weights('my_weights_tensorflow.h5')
#
# pdb.set_trace()
#
# k_model = pytorch_to_keras((20, 50, 50), output, change_ordering=True)
# k_model.load_weights('my_weights_theano.h5')

# pdb.set_trace()
# input_keras_tf = input_np.transpose(0,2,3,1)
# keras_output = tf_dim_model.predict(input_keras_tf)
#
#
# error = np.max(pytorch_output - keras_output)
# print(error)
#
# pdb.set_trace()
# test_model = Sequential()
# test_model.add(InputLayer(input_shape=(2, 10, 1)))
# test_model.add(Permute((3,2,1)))
# test_model.add(Reshape((-1, 2, 2)))
# test_model.add(Permute((3,2,1)))
#
# data = np.arange(20).reshape(1,2,10,1)
# result = test_model.predict(data)
# print result
#

# Convert keras model to coreml model
pdb.set_trace()
import coremltools

coreml_model = coremltools.converters.keras.convert(tf_dim_model,input_names = 'image',image_input_names = 'image')
coreml_model.save('RepetitionCounting.mlmodel')

pdb.set_trace()

## B x H x (W*C) x 1 -> B x H x W x C と変換
data.transpose(0,3,2,1)[0,0,:,:2].reshape(1,5,2,2).transpose(0,3,2,1)

## 入力形式の変換
# pytorch 入力 (B x C x H x W)
input_np = np.random.uniform(0, 1, (1, 20, 50, 50))

# tf 入力 (B x H x W x C)
input_np.transpose(0,2,3,1)

# tf 1channel化 入力 (B x H x W*C x 1) 水平方向にチャネルを結合
# 大分無駄な変形をしている気がするが，，， transpose, reshape だけでやろうとすると難しい
tf_1ch_input = input_np.transpose(0,1,3,2).reshape(1,1,1,-1).transpose(0,1,3,2).reshape(1,1,-1,50).transpose(0,3,2,1)

keras_output = tf_dim_model.predict(tf_1ch_input)
