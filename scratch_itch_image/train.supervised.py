import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import sys
import numpy as np
import os
from itertools import product

from models import make_TCN

parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--data_file',help='name of data file')
parser.add_argument('--log_dir', help='name of directory to log to')
args = parser.parse_args()

class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, batch_size=50, shuffle=True):
		'Initialization'
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.shuffle = shuffle
		self.on_epoch_end()
		self.cache = {}

	def __len__(self):
		'Denotes the number of batches per epoch'
		return len(self.list_IDs)*500//self.batch_size

	def __getitem__(self, index):
		'Generate one batch of data'
		X, Y = self.__data_generation(self.indexes[self.count:self.count+self.batch_size//10])
		self.count += self.batch_size//10

		return X, Y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs)*50)
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
		self.count = 0
		self.cache = {}

	def __data_generation(self, indices):
		'Generates data containing batch_size samples'
		X = np.empty((self.batch_size, 201, 27))
		Y = np.empty((self.batch_size, 200, 3))

		for index,count in zip(indices,range(0,len(indices*10),10)):
			file_id,block = index//50, index%50
			if file_id in self.cache:
				X[count:count+10] = self.cache[file_id][block*10:(block+1)*10]
			else:
				self.cache[file_id] = np.load(self.list_IDs[file_id])
				X[count:count+10] = self.cache[file_id][block*10:(block+1)*10]
		if len(self.cache) > 100: 
			self.cache.popitem()
		X = X[:,:-1,:]
		Y = np.tile(np.expand_dims(X[:,-1,:3],axis=1),(1,200,1))

		return X, Y

def serve_data(file):
	data = np.load(file)
	X,Y = data.values()
	Y = np.repeat(Y[:,np.newaxis,:],200,1)

	idx = np.random.choice(range(len(X)),int(.2*len(X)),replace=False)
	X_train,Y_train = np.delete(X,idx,axis=0),np.delete(Y,idx,axis=0)
	X_valid,Y_valid = X[idx],Y[idx]

	X_train = (X_train-np.mean(X_train,axis=1,keepdims=True))/np.std(X_train,axis=1,keepdims=True)
	X_valid = (X_valid-np.mean(X_valid,axis=1,keepdims=True))/np.std(X_valid,axis=1,keepdims=True)

	train_data = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
	train_data = train_data.shuffle(BUFFER_SIZE).cache().batch(BATCH_SIZE)

	val_data = tf.data.Dataset.from_tensor_slices((X_valid,Y_valid))
	val_data = val_data.batch(BATCH_SIZE)

	return train_data,val_data

BUFFER_SIZE = 1000
BATCH_SIZE = 400
EPOCHS = 50

if __name__ == "__main__":
	tf.debugging.set_log_device_placement(True)
	tf.keras.backend.clear_session()
	tf.config.optimizer.set_jit(True)


	# idx = np.random.choice(range(len(filenames)),int(.1*len(filenames)),replace=False)
	# train_files = np.delete(filenames,idx)
	# val_files = np.array(filenames)[idx]
	# train_data = DataGenerator(train_files,batch_size=500)
	# val_data = DataGenerator(val_files,batch_size=500)
	
	train_data,val_data = serve_data(args.data_file)
	print("datasets created")
	
	lrs = [1e-3,3e-3,5e-3]
	dropout = [.1,.2,.3,.4,.5]
	num_blocks = range(1,6)
	block_sizes = range(3,6)
	kernel_sizes = range(10,40,10)

	for lr,num_block,block_size,kernel_size in product(lrs,num_blocks,block_sizes,kernel_sizes):
		# model = make_TCN(lr,num_blocks=num_block,block_size=block_size,kernel_size=kernel_size)

		strategy = tf.distribute.MirroredStrategy()
		with strategy.scope():
			model = make_TCN(lr,num_blocks=num_block,block_size=block_size,kernel_size=kernel_size)
		
		model_dir = f'block_{num_block}_{block_size}_kernel_{kernel_size}_lr_{lr:.2E}'

		log_dir = os.path.join(args.log_dir,"logs",model_dir)
		save_path = os.path.join(args.log_dir,"checkpoints",model_dir)
		os.makedirs(log_dir, exist_ok=True)
		os.makedirs(save_path, exist_ok=True)
		
		checkpoint_path = save_path+"/epoch_{epoch:02d}.h5"
		model.save_weights(checkpoint_path.format(epoch=0))
		callbacks = [ModelCheckpoint(checkpoint_path,save_best_only=True,period=10),
					TensorBoard(log_dir,write_graph=False)]

		model.fit(train_data, epochs=EPOCHS, steps_per_epoch=None,
												verbose=2,
												callbacks=callbacks,
												validation_data=val_data,
												validation_steps=None,)
		model.save(save_path+"/final_model.h5")
