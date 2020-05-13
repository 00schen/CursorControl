import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from itertools import product
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam 

from models import make_TCN,TCN,make_LSTM,make_GRU
from utils import serve_data

parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--data_file',help='name of data file')
parser.add_argument('--log_dir', help='name of directory to log to')
parser.add_argument('--model',help='model to train on')
# parser.add_argument('--lr', default=1e-3, help='name of directory to log to')
args = parser.parse_args()

BUFFER_SIZE = 1000
BATCH_SIZE = 200
EPOCHS = 50

if __name__ == "__main__":
	tf.compat.v1.disable_eager_execution()
	tf.config.optimizer.set_jit(False) 

	# idx = np.random.choice(range(len(filenames)),int(.1*len(filenames)),replace=False)
	# train_files = np.delete(filenames,idx)
	# val_files = np.array(filenames)[idx]
	# train_data = DataGenerator(train_files,batch_size=500)
	# val_data = DataGenerator(val_files,batch_size=500)
	
	train_data,val_data = serve_data(args.data_file,BATCH_SIZE,BUFFER_SIZE)
	print("datasets created")
	
	model_types = {'lstm':make_LSTM,'gru':make_GRU,'tcn':make_TCN}
	lrs = [1e-3,3e-3,5e-3]
	dropouts = [0,.1,.2,.3,.4,.5]
	rdropouts = [0,.1,.2,.3,.4,.5]
	# num_blocks = range(5,1,-1)
	# block_sizes = range(5,2,-1)
	# kernel_sizes = range(10,40,10)

	num_layers = range(1,6)
	add_denses = [True,False]

	for dropout,rdropout,num_layer,add_dense,lr in product(dropouts,rdropouts,num_layers,add_denses,lrs):
		# lr = float(args.lr)

		# model = make_TCN(num_blocks=num_block,block_size=block_size,kernel_size=kernel_size,dropout_rate=dropout)
		model = model_types[args.model](dropout=dropout,recurrent_dropout=rdropout,layers=num_layer,add_dense=add_dense)
		# model.compile(optimizer=Adam(lr),loss='mse',run_eagerly=True)
		model.compile(optimizer=Adam(lr),loss='mse')
		
		# model_dir = f'block_{num_block}_{block_size}_kernel_{kernel_size}_drop_{dropout}_lr_{lr:.2E}'
		model_dir = f'model_{args.model}_drop_{dropout}_rdrop_{rdropout}_dense_{int(add_dense)}_lr_{lr:.2E}'

		log_dir = os.path.join(args.log_dir,"logs",model_dir)
		save_path = os.path.join(args.log_dir,"checkpoints",model_dir)
		os.makedirs(log_dir, exist_ok=True)
		os.makedirs(save_path, exist_ok=True)
		
		checkpoint_path = save_path+"/epoch_{epoch:02d}"
		callbacks = [ModelCheckpoint(checkpoint_path,save_best_only=True,period=10),
					TensorBoard(log_dir,write_graph=False)]

		model.fit(train_data, epochs=EPOCHS, verbose=2,
											callbacks=callbacks,
											validation_data=val_data,)
		model.save(save_path+"/final_model")
