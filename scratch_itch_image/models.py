import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from functools import reduce

class TCNBlock(tf.keras.layers.Layer):
    def __init__(self,channels=64,kernel_size=5,block_size=3,dilation=2,dropout_rate=0):
        super().__init__()
        self.layers = [layer() for layer in [
            lambda: Conv1D(channels,kernel_size, padding='causal',dilation_rate=dilation),
            lambda: LayerNormalization(),
            lambda: Activation('relu'),
            lambda: Dropout(dropout_rate),
            ] for _i in range(block_size)]
        self.shortcut = Conv1D(channels,1,padding='causal')
        self.add = Add()
        self.activation = Activation('relu')
        self.config = {
            "channels": channels,
            "kernel_size": kernel_size,
            "block_size": block_size,
            "dilation": dilation,
            "dropout_rate": dropout_rate,
        }

    # @tf.function
    def call(self,X,training=False):
        X_short = self.shortcut(X)
        for layer in self.layers:
            X_short = layer(X_short,training=training)
        return self.activation(self.add([X,X_short]))  

    def get_config(self):
        return self.config

def make_TCN(input_shape,channels=64,num_blocks=1,kernel_size=5,block_size=3,dilation=2,dropout_rate=0):
    model = tf.keras.Sequential([Conv1D(channels,1, padding='causal',input_shape=input_shape)]\
        + [TCNBlock(channels,kernel_size=kernel_size,block_size=block_size,dilation=dilation,dropout_rate=dropout_rate) for _i in range(num_blocks)]\
        + [Conv1D(3,kernel_size,padding='causal')]
        )
    
    return model

def make_LSTM(input_shape,dropout=0,recurrent_dropout=0,num_layer=2,channels=64,kernel_size=1):
    model = tf.keras.Sequential()
    model.add(LSTM(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True,input_shape=input_shape))
    for _i in range(num_layer-1):
        model.add(LSTM(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True))
    model.add(Conv1D(3,kernel_size,padding='causal'))
    return model

def make_GRU(input_shape,dropout=0,recurrent_dropout=0,num_layer=2,channels=64,kernel_size=1):
    model = tf.keras.Sequential()
    model.add(GRU(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True,input_shape=input_shape))
    for _i in range(num_layer-1):
        model.add(GRU(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True))
    model.add(Conv1D(3,kernel_size,padding='causal'))
    return model