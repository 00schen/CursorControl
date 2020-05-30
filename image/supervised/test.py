import tensorflow as tf
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM

X,Y = np.random.random((int(5e4),200,27)),np.random.random((int(5e4),200,3))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

model = tf.keras.Sequential()
model.add(LSTM(3,return_sequences=True,input_shape=(200,27)))
model.compile(optimizer=Adam(1e-3),loss='mse',run_eagerly=False)

model.fit(X,Y)