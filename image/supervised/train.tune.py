import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam

from models import make_TCN,make_LSTM,make_LSTM1,make_LSTM2
from utils import *
from tqdm import tqdm
import gym

dirname = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
parser.add_argument('--data_file',help='name of data file')
parser.add_argument('--local_dir',help='dir to save trials')
parser.add_argument('--env_name',help='gym environment name')
parser.add_argument('--exp_name',help='experiment name')
parser.add_argument('--model',help='model to use')
parser.add_argument('--oracle',default='trajectory',help='oracle to use')
parser.add_argument('--num_gpu',default=4,help='number of visible gpus')
args, _ = parser.parse_known_args()

DATA_SIZE = 512
BUFFER_SIZE = 1000
BATCH_SIZE = 64
EPOCHS = 50

class AssistiveTrainable(tune.Trainable):
    def _setup(self, config):
        import tensorflow as tf
        # from tensorflow.compat.v1 import ConfigProto,Session
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            pass
        # tf_config = ConfigProto()
        # tf_config.gpu_options.allow_growth = True
        # session = Session(config=tf_config)

        # self.train_data,self.valid_data = serve_data(args.data_file,config['input_shape'],BATCH_SIZE,BUFFER_SIZE)
        obs,targets,labels = serve_data(args.data_file,DATA_SIZE)
        self.data = Data(obs,targets,labels,.2)
        
        self.reset_config(config)
        self.samplers = [Sampler.remote(args.env_name,1000+i,config['oracle'])\
             for i in range(config['num_samplers'])]

    def reset_config(self, config):
        from tensorflow.keras.optimizers import RMSprop, Adam, Nadam

        self.data.reset()

        model = self.model_name = config['model']
        input_shape = config['input_shape']
        # if model == "tcn":
        #     lr,num_channel,num_block,block_size,kernel_size,dropout = config['lr'],config['num_channel'],config['num_block'],config['block_size'],config['kernel_size'],config['dropout']
        #     self.model = make_TCN(
        #                             input_shape,
        #                             channels=num_channel,
        #                             num_blocks=num_block,
        #                             block_size=block_size,
        #                             kernel_size=(kernel_size,),
        #                             dropout_rate=dropout)
        #     self.model.compile(optimizer=Adam(lr),loss='mse',run_eagerly=False)
        if model == "lstm":
            dropout,rdropout,num_layer,lr = config['dropout'],config['rdropout'],config['num_layer'],config['lr']
            self.model = make_LSTM(
                                    input_shape,
                                    dropout=dropout,
                                    recurrent_dropout=rdropout,
                                    num_layer=num_layer)
            self.model.compile(optimizer=Adam(lr,clipvalue=1),loss='binary_crossentropy',run_eagerly=False,metrics=['binary_accuracy'])
        elif model == "lstm1":
            dropout,rdropout,num_layer,lr = config['dropout'],config['rdropout'],config['num_layer'],config['lr']
            self.model = make_LSTM1(
                                    input_shape,
                                    dropout=dropout,
                                    recurrent_dropout=rdropout,
                                    num_layer=num_layer)
            self.model.compile(optimizer=Adam(lr,clipvalue=1),
                                loss={'target':'mse','label':'binary_crossentropy'},
                                metrics={'label':'binary_accuracy'},
                                run_eagerly=False)
        elif model == "lstm2":
            dropout,rdropout,num_layer,lr = config['dropout'],config['rdropout'],config['num_layer'],config['lr']
            self.model = make_LSTM2(
                                    input_shape,
                                    dropout=dropout,
                                    recurrent_dropout=rdropout,
                                    num_layer=num_layer)
            self.model.compile(optimizer=Adam(lr,clipvalue=1),
                                loss={'target':'mse','label':'binary_crossentropy'},
                                metrics={'label':'binary_accuracy'},
                                run_eagerly=False)


    def _train(self):
        if self.model_name == 'lstm':
            self.model.fit(self.data['train']['X'],self.data['train']['C'],
                                                batch_size=BATCH_SIZE,
                                                epochs=1,
                                                verbose=2)
            results = self.model.evaluate(self.data['train']['X'],self.data['train']['C'],verbose=2)
        elif self.model_name == 'lstm1' or self.model_name == 'lstm2':
            self.model.fit(self.data['train']['X'],
                            {'target':self.data['train']['Y'],
                            'label':self.data['train']['C']},
                                                batch_size=BATCH_SIZE,
                                                epochs=1,
                                                verbose=2)
            results = self.model.evaluate(self.data['train']['X'],
                            {'target':self.data['train']['Y'],
                            'label':self.data['train']['C']},
                            verbose=2)
        results = dict(zip(self.model.metrics_names, results))

        self._save(self.logdir)
        samples = [sampler.sample.remote(DATA_SIZE//len(self.samplers),os.path.join(self.logdir,'model.h5')) for sampler in self.samplers]
        samples = [ray.get(sample) for sample in samples]
        os.remove(os.path.join(self.logdir,'model.h5'))
        obs,targets,labels = zip(*samples)
        obs,targets,labels = list(obs),list(targets),list(labels)
        obs,targets,labels = np.concatenate(obs),np.concatenate(targets),np.concatenate(labels)
        # obs,targets = np.random.random((512,100,27)),np.random.random((512,100,3))
        self.data.update(obs,targets,labels)
        np.savez_compressed(os.path.join(self.logdir,f'training_data_{self.iteration}'),obs=obs,targets=targets,labels=labels)

        return results

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.h5")
        self.model.save(checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        from tensorflow.keras.models import load_model
        self.model = load_model(checkpoint_path)

@ray.remote(num_cpus=1)
class Sampler:
    def __init__(self,env_name,seed,oracle):
        env_name0 = env_name[:-1]+'0'
        env_name1 = env_name[:-1]+'1'
        env = self.env = gym.make(env_name1)
        env.seed(seed)
        pretrain = PretrainAgent(os.path.join(dirname,'trained_models','ppo',env_name0+'.pt'))
        if oracle == 'trajectory':
            self.oracle = TrajectoryOracle(pretrain,env.oracle2trajectory,env_name1)
        elif oracle == 'target':
            self.oracle = TargetOracle(pretrain,env,env_name1)
        pretrain.add()
        self.agent = BufferAgent(pretrain,Predictor('twin',oracle_type=oracle),env.target2obs)

    def sample(self,count,predictor_path):
        predictor = tf.keras.models.load_model(predictor_path)

        targets = []
        obs_data = []
        labels = []

        env,oracle,agent = self.env,self.oracle,self.agent
        for _i in tqdm(range(count)):
            target_ep = []
            obs_ep = []

            obs = env.reset()
            oracle.reset()
            obs = oracle.predict(obs)

            label_ep = [env.label]*100

            agent.reset(predictor)
            action = agent.predict(obs)
            for i in range(100):
                target_ep.append(agent.predictor.target_norm(env.target_pos))
                obs_ep.append(agent.predictor.norm(np.concatenate((*obs['obs'],obs['action']))))

                obs,_r,done,_info = env.step(action)
                obs = oracle.predict(obs,done)

                action = agent.predict(obs,done)
            
            targets.append(target_ep)
            obs_data.append(obs_ep)
            labels.append(label_ep)
        labels = tf.one_hot(labels,2).numpy()

        return obs_data,targets,labels

    def close(self):
        env.close()

if __name__ == "__main__":
    ray.init(num_gpus=int(args.num_gpu),temp_dir='/tmp/ray_exp')

    if args.env_name == "ScratchItchJaco-v0":
        input_shape = (100,27)
    elif args.env_name == "FeedingJaco-v0":
        input_shape = (100,25)

    # tcn_sched = AsyncHyperBandScheduler(
    #     time_attr="training_iteration",
    #     metric="loss",
    #     mode="min",
    #     max_t=50,
    #     grace_period=30)
    lstm_sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=60,
        grace_period=40)

    # tcn_space = {
    #     "lr": hp.uniform("lr",1e-3, 5e-3),
    #     "num_channel": hp.randint("num_channel",64,256),
    #     "dropout": hp.uniform("dropout",0.1, 0.5),
    #     "num_block": hp.randint("num_block",2, 6),
    #     "block_size": hp.randint("block_size",2, 6),
    #     "kernel_size": hp.randint("kernel_size",10, 50),
    # }
    # tcn_algo = HyperOptSearch(
    #     tcn_space,
    #     metric="loss",
    #     mode="min",)

    lstm_space = {
        "lr": hp.uniform("lr",1e-3, 1e-2),
        "dropout": hp.uniform("dropout",0, 0.5),
        # "rdropout": hp.uniform("rdropout",0, 0.5),
        "num_layer": hp.randint("num_layer",1,5),
        "model": hp.choice("model",['lstm1','lstm2'])
    }
    current_best_params = [
    {
    'lr': 5e-3,
    'dropout': 0.05,
    'num_layer': 2,
    'model': 0,
    },
    {
    'lr': 5e-3,
    'dropout': 0.05,
    'num_layer': 2,
    'model': 1,
    },
    ]
    lstm_algo = HyperOptSearch(
        lstm_space,
        metric="loss",
        mode="min",
        points_to_evaluate=current_best_params)

    # tcn_exp = tune.Experiment(
    #                     args.exp_name,
    #                     AssistiveTrainable,
    #                     local_dir=args.local_dir,
    #                     stop={"training_iteration":50},
    #                     resources_per_trial={
    #                         "cpu": 1,
    #                         "gpu": 1
    #                     },
    #                     num_samples=300,
    #                     checkpoint_freq=10,
    #                     checkpoint_at_end=True,
    #                     config = {
    #                         "model":"tcn",
    #                         "input_shape":input_shape,
    #                         })

    lstm_exp = tune.Experiment(
                        args.exp_name,
                        AssistiveTrainable,
                        local_dir=args.local_dir,
                        stop={"training_iteration":100},
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": 1 if int(args.num_gpu) else 0,
                            "extra_cpu":16,
                            # "extra_gpu":2,
                        },
                        num_samples=60,
                        checkpoint_freq=2,
                        checkpoint_at_end=True,
                        config = {
                            # "model":args.models,
                            "input_shape": input_shape,
                            "oracle": args.oracle,
                            "num_samplers": 16,
                            "rdropout": 0,
                            })

    # if args.model == "tcn":
    #     tune.run_experiments(
    #         tcn_exp,
    #         reuse_actors=True,
    #         search_alg=tcn_algo,
    #         scheduler=tcn_sched,
    #         verbose=1,
    #     )
    tune.run_experiments(
        lstm_exp,
        reuse_actors=True,
        search_alg=lstm_algo,
        scheduler=lstm_sched,
        verbose=1,
    )