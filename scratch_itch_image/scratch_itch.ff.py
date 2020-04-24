import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import sys
import numpy as np
import os

parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--compute', action='store_false',
                    help='use multiprocessing (default: true)')
args = parser.parse_args()

BATCH_SIZE = 256
BUFFER_SIZE = 10000
EPOCHS = 10
EVALUATION_INTERVAL = 500
train_filenames = ['12345/3.tfrecord','12345/4.tfrecord','12345/5.tfrecord',
			'123456/1.tfrecord','123456/2.tfrecord','123456/3.tfrecord','123456/4.tfrecord','123456/5.tfrecord',
			'123456/6.tfrecord','123456/7.tfrecord','123456/8.tfrecord','123456/9.tfrecord','123456/10.tfrecord','123456/11.tfrecord',
			'1234567/1.tfrecord','1234567/2.tfrecord','1234567/3.tfrecord','1234567/4.tfrecord',]
valid_filenames = ['12345/1.tfrecord','12345/2.tfrecord',]

# Create a description of the features.
feature_description = {
		'lstm_input': tf.io.FixedLenFeature([], tf.string, default_value=''),
		'target': tf.io.FixedLenFeature([], tf.string, default_value=''),
	}

def _parse_function(example_proto):
	# Parse the input `tf.Example` proto using the dictionary above.
	parsed_ex = tf.io.parse_single_example(example_proto, feature_description)
	parsed_ex['lstm_input'] = tf.io.parse_tensor(parsed_ex['lstm_input'],out_type=tf.float64)
	parsed_ex['lstm_input'].set_shape((500,31))
	parsed_ex['target'] = tf.io.parse_tensor(parsed_ex['target'],out_type=tf.float64)
	parsed_ex['target'].set_shape((1,3))
	parsed_ex['target'] = tf.tile(tf.expand_dims(parsed_ex['target'],1),tf.constant([1,500,1], tf.int32))
	return parsed_ex

def load_data(filenames):
	raw_dataset = tf.data.TFRecordDataset(filenames)
	parsed_dataset = raw_dataset.map(_parse_function)
	return parsed_dataset

class FeedForward(tf.keras.Model):
	def __init__(self, rollout, input_size):
		super(FeedForward, self).__init__()
		self.rollout = rollout
		self.net = tf.keras.models.Sequential()
		self.net.add(tf.keras.layers.Dense(32,input_shape=(rollout,input_size,)))
		self.net.add(tf.keras.layers.Dense(32))
		self.net.add(tf.keras.layers.Dense(3))

	def forward(self, x):
		x = tf.concat([tf.tile(x[:,0,:],tf.constant([1,self.rollout-1,1],tf.int32)),x],axis=1)
		return np.array([self.net(x[:,i:i+self.rollout,:])\
			for i in range(x.shape[1]-self.rollout)])


if __name__ == "__main__":
	save_path = 'scratch_itch_image/lstm/'
	final_model_path = 'scratch_itch_image/lstm/1/'
	os.makedirs(save_path, exist_ok=True)
	os.makedirs(final_model_path, exist_ok=True)

	train_data = load_data(train_filenames)
	train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

	val_data = load_data(valid_filenames)
	val_data = val_data.cache().batch(BATCH_SIZE)

	model = FeedForward(5,(500,31))

	model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mse')

	model.fit(train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
											callbacks=[ModelCheckpoint(save_path),TensorBoard()],
											validation_data=val_data,
											validation_steps=50)
											# use_multiprocessing=bool(args.compute))
	tf.saved_model.save(model,final_model_path)
