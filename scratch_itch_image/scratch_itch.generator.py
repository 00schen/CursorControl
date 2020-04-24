import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam 
import sys
import numpy as np
import os
from itertools import product

from model_maps import conv1d_model

parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--log_dir', help='name of directory to log to')
args = parser.parse_args()

filenames = [os.path.join("scratch_itch","12345_"+str(file_num)+".npy") for file_num in range(10,440+10,10)]\
	+[os.path.join("scratch_itch","123456_"+str(file_num)+".npy") for file_num in range(10,1030+10,10)]\
	+[os.path.join("scratch_itch","1234567_"+str(file_num)+".npy") for file_num in range(10,790+10,10)]

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
		X = np.empty((self.batch_size, 201, 31))
		Y = np.empty((self.batch_size, 200, 3))

		for index,count in zip(indices,range(0,len(indices*10),10)):
			file_id,block = index//50, index%50
			if file_id in self.cache:
				X[count:count+10] = self.cache[file_id][block*10:(block+1)*10]
			else:
				self.cache[file_id] = np.load(self.list_IDs[file_id])
				X[count:count+10] = self.cache[file_id][block*10:(block+1)*10]
		if len(self.cache) > len(self.list_IDs)//2: 
			self.cache.popitem()
		X = X[:,:-1,:]
		Y = np.tile(np.expand_dims(X[:,-1,:3],axis=1),(1,200,1))

		return X, Y

# def make_ds_generator(filelist):
# 	def dataset_generator():
# 		for file in filelist:
# 			data = np.load(file)
# 			X = data[:,:-1,:]
# 			Y = np.tile(np.expand_dims(X[:,-1,:3],axis=1),(1,200,1))
# 			return (X,Y)

# 	return dataset_generator

def serve_data(filelist):
	data = []
	count = 0
	for file in filelist:
		try:
			new_data = np.load(file)
		except:
			continue
		data = np.concatenate((data,new_data), axis=0) if len(data) else new_data
	
	X = data
	X = X[:,:-1,:]
	Y = np.tile(np.expand_dims(X[:,-1,:3],axis=1),(1,200,1))

	idx = np.random.choice(range(len(X)),int(.1*len(X)),replace=False)
	X_train,Y_train = np.delete(X,idx),np.delete(Y,idx)
	X_valid,Y_valid = X[idx],Y[idx]

	train_data = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
	train_data = train_data.cache().shuffle().batch(BATCH_SIZE)

	val_data = tf.data.Dataset.from_tensor_slices((X_valid,Y_valid))
	val_data = val_data.cache().batch(BATCH_SIZE)

	return train_data,val_data

BATCH_SIZE = 256
BUFFER_SIZE = 10000
EPOCHS = 100
EVALUATION_INTERVAL = 500

if __name__ == "__main__":
	idx = np.random.choice(range(len(filenames)),int(.1*len(filenames)),replace=False)
	train_files = np.delete(filenames,idx)
	val_files = np.array(filenames)[idx]
	train_data = DataGenerator(train_files,batch_size=500)
	val_data = DataGenerator(val_files,batch_size=500)

	# train_data,val_data = serve_data(filenames)

	num_layers = range(2,5)
	losses = ['mse','cosine_similarity','huber']
	optimizers = {'RMSprop':RMSprop,'Adam':Adam,'Nadam':Nadam}
	lrs = np.geomspace(1e-4,.1,8)

	for num_layer,loss,opt,lr in product(num_layers,losses,optimizers,lrs):
		model = conv1d_model(num_layer,optimizers[opt](lr),loss)

		model_dir = f'layer_{num_layer}_loss_{loss}_opt_{opt}_lr_{lr}'
		log_dir = os.path.join(args.log_dir,model_dir,"logs")
		save_path = os.path.join(args.log_dir,model_dir,"checkpoints")
		final_model_path = os.path.join(args.log_dir,model_dir)
		os.makedirs(log_dir, exist_ok=True)
		os.makedirs(save_path, exist_ok=True)
		os.makedirs(final_model_path, exist_ok=True)
		
		checkpoint_path = save_path+"/epoch_{epoch:02d}.ckpt"
		model.save_weights(checkpoint_path.format(epoch=0))
		callbacks = [ModelCheckpoint(checkpoint_path,save_weights_only=True,period=5),TensorBoard(log_dir)]

		model.fit(train_data, epochs=EPOCHS, steps_per_epoch=None,
												callbacks=callbacks,
												validation_data=val_data,
												validation_steps=None,
												workers=1,
												shuffle=True)
		model.save(final_model_path+"/final_model.h5")
