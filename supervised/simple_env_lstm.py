import argparse
import tensorflow as tf
import sys
from utils import data_loader
import numpy as np
import os

parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='velocitycontrol-v0',
                    help='the dataset to run (default: velocitycontrol-v0)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

BATCH_SIZE = 256
BUFFER_SIZE = 10000
EPOCHS = 100
EVALUATION_INTERVAL = 30

if __name__ == "__main__":
    args = parser.parse_args()
    save_path = 'simple_env/lstm/1/'
    os.makedirs(save_path, exist_ok=True)

    print(args)
    X_train, X_valid = data_loader(args.data)
    # X_train, X_valid = X_train[:len(X_train)//10], X_valid[:len(X_valid)//10]
    X_train = X_train[:,:-1,:]
    Y_train = np.tile(X_train[:,-1,:2].reshape((-1,1,2)),(1,X_train.shape[1],1))
    X_valid = X_valid[:,:-1,:]
    Y_valid = np.tile(X_valid[:,-1,:2].reshape((-1,1,2)),(1,X_valid.shape[1],1))

    train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(32,input_shape=X_train.shape[-2:],return_sequences=True))
    model.add(tf.keras.layers.Dense(2))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mse')

    model.fit(train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data,
                                            validation_steps=30)
    tf.saved_model.save(model,save_path)

    