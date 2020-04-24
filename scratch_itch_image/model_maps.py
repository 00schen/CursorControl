import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

def conv1d_model(num_layers,optimizer,loss):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(32,5, padding='causal', input_shape=(200,31)))
    for _i in range(num_layers-2):
        model.add(tf.keras.layers.Conv1D(32,5, padding='causal'))
    model.add(tf.keras.layers.Conv1D(3,5, padding='causal', activation='relu'))
    model.compile(optimizer=optimizer, loss=loss)
    return model
