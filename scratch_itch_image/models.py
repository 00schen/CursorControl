import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam 
from functools import reduce

class TCNBlock(tf.keras.layers.Layer):
    def __init__(self,kernel_size=5,block_size=3,dilation=2,dropout_rate=0):
        super().__init__()
        self.layers = []
        for _i in range(block_size-1):
            self.layers.extend([Conv1D(64,kernel_size, padding='causal',dilation_rate=dilation),
                        LayerNormalization(),
                        Activation('relu'),
                        Dropout(dropout_rate),
            ])
        self.shortcut = Conv1D(64,1,padding='causal')
        self.add = Add()
        self.activation = Activation('relu')

    def call(self,X,trainable=False):
        X_short = self.shortcut(X)
        X = reduce(lambda value,func: func(value),self.layers,X)
        return self.add([X,X_short])       

def make_TCN(lr,num_blocks=1,kernel_size=5,block_size=3,dilation=2,dropout_rate=0):
    model = tf.keras.Sequential([Conv1D(64,1, padding='causal',input_shape=(200,27))]\
        + [TCNBlock(kernel_size=kernel_size,block_size=block_size,dilation=dilation,dropout_rate=dropout_rate) for _i in range(num_blocks)]\
        + [Conv1D(3,kernel_size,padding='causal')]
        )
    model.compile(optimizer=Adam(lr),loss='mse')
    return model

