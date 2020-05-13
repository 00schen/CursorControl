import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam 
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
            "drop_rate": dropout_rate,
        }

    # @tf.function
    def call(self,X,training=False):
        X_short = self.shortcut(X)
        for layer in self.layers:
            X_short = layer(X_short,training=training)
        return self.activation(self.add([X,X_short]))  

    def get_config(self):
        return self.config

class TCN(tf.keras.Model):
    def __init__(self,channels=64,num_blocks=1,kernel_size=5,block_size=3,dilation=2,dropout_rate=0):
        super().__init__()
        self.start = Conv1D(channels,1, padding='causal',input_shape=(200,27))
        self.blocks = [
            TCNBlock(kernel_size=kernel_size,block_size=block_size,dilation=dilation,dropout_rate=dropout_rate)\
            for _i in range(num_blocks)]
        self.classifier = Conv1D(3,kernel_size,padding='causal')

    # @tf.function
    def call(self,X,training=False):
        X = self.start(X)
        for block in self.blocks:
            X = block(X,training=training)
        return self.classifier(X)

def make_TCN(channels=64,num_blocks=1,kernel_size=5,block_size=3,dilation=2,dropout_rate=0):
    model = tf.keras.Sequential([Conv1D(channels,1, padding='causal',input_shape=(200,27))]\
        + [TCNBlock(channels,kernel_size=kernel_size,block_size=block_size,dilation=dilation,dropout_rate=dropout_rate) for _i in range(num_blocks)]\
        + [Conv1D(3,kernel_size,padding='causal')]
        )
    
    return model

def make_LSTM(channels=64,dropout=0,recurrent_dropout=0,layers=1,add_dense=False):
    model = tf.keras.Sequential()
    if layers > 1:
        model.add(LSTM(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True,input_shape=(200,27)))
        for _i in range(layers-2):
            model.add(LSTM(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True))
     
        if not add_dense:
            model.add(LSTM(3,dropout=dropout,recurrent_dropout=recurrent_dropout,activation=None,return_sequences=True))
            return model
        else:
            model.add(LSTM(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True))
            model.add(Conv1D(3,20,padding='causal'))
            return model 
    elif layers == 1 and not add_dense:
        model.add(LSTM(3,dropout=dropout,recurrent_dropout=recurrent_dropout,activation=None,return_sequences=True,input_shape=(200,27)))
        return model
    else:
        model.add(LSTM(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=True,input_shape=(200,27)))
        model.add(Conv1D(3,20,padding='causal'))
        return model

def make_GRU(channels=64,dropout=0,recurrent_dropout=0,layers=1,add_dense=False):
    model = tf.keras.Sequential()
    if layers > 1:
        model.add(GRU(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,reset_after=True,return_sequences=True,input_shape=(200,27)))
        for _i in range(layers-2):
            model.add(GRU(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,reset_after=True,return_sequences=True))
     
        if not add_dense:
            model.add(GRU(3,dropout=dropout,recurrent_dropout=recurrent_dropout,reset_after=True,activation=None,return_sequences=True))
            return model
        else:
            model.add(GRU(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,reset_after=True,return_sequences=True))
            model.add(Conv1D(3,20,padding='causal'))
            return model 
    elif layers == 1 and not add_dense:
        model.add(GRU(3,dropout=dropout,recurrent_dropout=recurrent_dropout,reset_after=True,activation=None,return_sequences=True,input_shape=(200,27)))
        return model
    else:
        model.add(GRU(channels,dropout=dropout,recurrent_dropout=recurrent_dropout,reset_after=True,return_sequences=True,input_shape=(200,27)))
        model.add(Conv1D(3,20,padding='causal'))
        return model