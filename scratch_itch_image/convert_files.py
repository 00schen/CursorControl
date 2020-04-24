import argparse
import os
import sys

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description='Write npy files as TfRecords')
parser.add_argument('--file_dir', type=str, default='', help='file directory')
parser.add_argument('--file_head', type=str, default='', help='header of file names')
parser.add_argument('--start', type=int, help='starting file number')
parser.add_argument('--end', type=int, help='ending file number')
parser.add_argument('--stride', type=int, help='number of samples to write to single record')
args = parser.parse_args()

save_dir = "scratch_itch/"
os.makedirs(save_dir, exist_ok=True)

if __name__ == '__main__':
	data = []
	save_count = 0
	for file_num in range(args.start, args.end+10, 10):
		try:
			new_data = np.load(args.file_dir+'/'+args.file_head+str(file_num)+'.npy')
		except:
			break
		data = np.concatenate((data,new_data), axis=0) if len(data) else new_data
		if not data.shape[0] % args.stride:
			data = data[:,np.concatenate((np.arange(200),[-1])),:]
			save_count += 10
			np.save(save_dir+args.file_dir+'_'+str(save_count), data)
			data = []

# def _bytes_feature(value):
# 	"""Returns a bytes_list from a string / byte."""
# 	if isinstance(value, type(tf.constant(0))):
# 		value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
# 	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def serialize_example(lstm_input, target):
# 	"""
# 	Creates a tf.Example message ready to be written to a file.
# 	"""
# 	# Create a dictionary mapping the feature name to the tf.Example-compatible
# 	# data type.
# 	feature = {
# 		'observation': _bytes_feature(tf.io.serialize_tensor(lstm_input)),
# 		'target': _bytes_feature(tf.io.serialize_tensor(target)),
# 	}

# 	# Create a Features message using tf.train.Example.

# 	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
# 	return example_proto.SerializeToString()

# def tf_serialize_example(lstm_input,target):
# 	"""Wraps serialize to return tensor."""
# 	tf_string = tf.py_function(
# 		serialize_example,
# 		(lstm_input,target),  # pass these args to the above function.
# 		tf.string)      # the return type is `tf.string`.
# 	return tf.reshape(tf_string, ()) # The result is a scalar


# def serve_dataset(file_head,start,end):
# 	"""Load numpy array and returns as tf Dataset"""
# 	data = []
# 	for file_num in range(start, end, 10):
# 		try:
# 			new_data = np.load(file_head+str(file_num)+'.npy')
# 		except:
# 			break
# 		# print(new_data.shape)
# 		data = np.concatenate((data,new_data), axis=0) if len(data) else new_data
# 	X = data[:,:-1,:]
# 	Y = X[:,-1,:3]
# 	dataset = tf.data.Dataset.from_tensor_slices((X,Y))
# 	return dataset

# def write_dataset(file_name,dataset):
# 	serialized_dataset = dataset.map(tf_serialize_example)
# 	writer = tf.data.experimental.TFRecordWriter(file_name)
# 	writer.write(serialized_dataset)

# if __name__ == '__main__':
# 	for index in range(int(np.ceil(args.end/args.stride))):
# 		file_head = args.file_dir+"scratch_itch_data_"
# 		np_dataset = serve_dataset(file_head,args.start+index*args.stride,args.start+(index+1)*args.stride)
# 		write_dataset(args.file_dir+str(index+1)+".tfrecord",np_dataset)
