import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
import numpy as np

from models import make_TCN,TCN,make_LSTM,make_GRU
from utils import serve_data
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam


parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
parser.add_argument('--data_file',help='name of data file')
parser.add_argument('--local_dir',help='dir to save trials')
parser.add_argument('--exp_name',help='experiment name')
parser.add_argument('model',help='model to use')
args, _ = parser.parse_known_args()

BUFFER_SIZE = 1000
BATCH_SIZE = 200
EPOCHS = 50

class ScratchItchTrainable(tune.Trainable):
    def _setup(self, config):
        # self.load_model = tf.keras.models.load_model

        # data_dir = os.path.join('/home/00schen/share','scratch_itch_image',args.data_file)
        # print(data_dir)
        self.train_data,self.valid_data = serve_data(args.data_file,BATCH_SIZE,BUFFER_SIZE)
        
        model = config['model']
        if model == "tcn":
            lr,num_channel,num_block,block_size,kernel_size,dropout = config['lr'],config['num_channel'],config['num_block'],config['block_size'],config['kernel_size'],config['dropout']
            self.model = make_TCN(channels=num_channel,
                                    num_blocks=num_block,
                                    block_size=block_size,
                                    kernel_size=(kernel_size,),
                                    dropout_rate=dropout)
            self.model.compile(optimizer=Adam(lr),loss='mse')
        elif model == "lstm":
            num_channel,dropout,rdropout,num_layer,add_dense,lr = config['num_channel'],config['dropout'],config['rdropout'],config['num_layer'],config['add_dense'],config['lr']
            self.model = make_LSTM(channels=num_channel,
                                    dropout=dropout,
                                    recurrent_dropout=rdropout,
                                    layers=num_layer,
                                    add_dense=add_dense)
            self.model.compile(optimizer=Adam(lr),loss='mse')
        elif model == "gru":
            num_channel,dropout,rdropout,num_layer,add_dense,lr = config['num_channel'],config['dropout'],config['rdropout'],config['num_layer'],config['add_dense'],config['lr']
            self.model = make_GRU(channels=num_channel,
                                    dropout=dropout,
                                    recurrent_dropout=rdropout,
                                    layers=num_layer,
                                    add_dense=add_dense)
            self.model.compile(optimizer=Adam(lr),loss='mse')

    def reset_config(self, config):
        model = config['model']
        if model == "tcn":
            lr,num_channel,num_block,block_size,kernel_size,dropout = config['lr'],config['num_channel'],config['num_block'],config['block_size'],config['kernel_size'],config['dropout']
            self.model = make_TCN(channels=num_channel,
                                    num_blocks=num_block,
                                    block_size=block_size,
                                    kernel_size=(kernel_size,),
                                    dropout_rate=dropout)
            self.model.compile(optimizer=Adam(lr),loss='mse')
        elif model == "lstm":
            num_channel,dropout,rdropout,num_layer,add_dense,lr = config['num_channel'],config['dropout'],config['rdropout'],config['num_layer'],config['add_dense'],config['lr']
            self.model = make_LSTM(channels=num_channel,
                                    dropout=dropout,
                                    recurrent_dropout=rdropout,
                                    layers=num_layer,
                                    add_dense=add_dense)
            self.model.compile(optimizer=Adam(lr),loss='mse')
        elif model == "gru":
            num_channel,dropout,rdropout,num_layer,add_dense,lr = config['num_channel'],config['dropout'],config['rdropout'],config['num_layer'],config['add_dense'],config['lr']
            self.model = make_GRU(channels=num_channel,
                                    dropout=dropout,
                                    recurrent_dropout=rdropout,
                                    layers=num_layer,
                                    add_dense=add_dense)
            self.model.compile(optimizer=Adam(lr),loss='mse')

    def _train(self):
        self.model.fit(self.train_data, epochs=1, validation_data=None,verbose=2)

        valid_loss = self.model.evaluate(self.valid_data,verbose=0)

        # It is important to return tf.Tensors as numpy objects.
        return {
            "val_loss": valid_loss,
        }

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.h5")
        self.model.save(checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model = tf.keras.models.load_model(checkpoint_path)

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.config.optimizer.set_jit(False) 
    ray.init(num_gpus=4,temp_dir='/tmp/ray_exp')

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="val_loss",
        mode="min",
        max_t=50,
        grace_period=20)

    tcn_space = {
        "lr": hp.uniform("lr",1e-3, 5e-3),
        "num_channel": hp.randint("num_channel",64,200),
        "dropout": hp.uniform("dropout",0.1, 0.5),
        "num_block": hp.randint("num_block",2, 6),
        "block_size": hp.randint("block_size",2, 6),
        "kernel_size": hp.randint("kernel_size",10, 50),
    }
    tcn_current_best_params = [
        {
            "lr": 1e-3,
            "num_channel": 64,
            "dropout": .1,
            "num_block": 5,
            "block_size": 5,
            "kernel_size": 30,
        }
    ]
    tcn_algo = HyperOptSearch(
        tcn_space,
        metric="val_loss",
        mode="min",
        points_to_evaluate=tcn_current_best_params)

    lstm_space = {
        "lr": hp.uniform("lr",1e-3, 1e-2),
        "num_channel": hp.randint("num_channel",64,256),
        "dropout": hp.uniform("dropout",0, 0.5),
        "rdropout": hp.uniform("rdropout",0, 0.5),
        "num_layer": hp.randint("num_layer",1, 6),
        "add_dense": hp.randint("add_dense",0,2),
    }
    lstm_algo = HyperOptSearch(
        lstm_space,
        metric="val_loss",
        mode="min")

    tcn_exp = tune.Experiment(
                        args.exp_name,
                        ScratchItchTrainable,
                        local_dir=args.local_dir,
                        stop={"training_iteration":30},
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": 1
                        },
                        num_samples=200,
                        checkpoint_freq=10,
                        checkpoint_at_end=True,
                        config = {"model":"tcn"})

    lstm_exp = tune.Experiment(
                        args.exp_name,
                        ScratchItchTrainable,
                        local_dir=args.local_dir,
                        stop={"training_iteration":30},
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": 1
                        },
                        num_samples=60,
                        checkpoint_freq=10,
                        checkpoint_at_end=True,
                        config = {"model":"lstm"})

    config={
            "lr": tune.sample_from(lambda spec: np.random.uniform(1e-3, 1e-2)),
            "dropout": tune.sample_from(lambda spec: np.random.uniform(.1,.5)),
            "num_block": tune.sample_from(lambda spec: np.random.randint(2,6)),
            "block_size": tune.sample_from(lambda spec: np.random.randint(3,6)),
            "kernel_size": tune.sample_from(lambda spec: np.random.randint(10,50)),
    }

    if args.model == "tcn":
        tune.run_experiments(
            tcn_exp,
            reuse_actors=True,
            search_alg=tcn_algo,
            # scheduler=sched,
            verbose=1,
        )
    elif args.model == "lstm":
        tune.run_experiments(
            lstm_exp,
            reuse_actors=True,
            search_alg=lstm_algo,
            # scheduler=sched,
            verbose=1,
        )