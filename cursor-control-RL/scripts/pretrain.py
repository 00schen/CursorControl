from __future__ import division
import pickle
import random
import uuid
import os

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

# gradient descent hyperparams
print_freq = 5000
iterations = 1000000
batch_size = 32
learning_rate = 1e-4
history_len = 1 # max number of timesteps into the past that the RNN decoder can look at during training
layer_size = 256
n_layers = 2

# hyperparams that don't need to be modified
n_act_dim = 2 # vx, vy
n_obs_dim = 128*7 # number of features in BCI output (using engineered features)

# make data directory for this session
session_id = 'pretrain'
curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, session_id)
if not os.path.exists(data_dir):
  os.makedirs(data_dir)

def save_tf_vars(scope, path):
  saver = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.save(sess, save_path=path)

# for RNN training
def build_mask(i, n):
  x = np.zeros(n)
  x[:i] = 1
  return x

pad_obses = lambda obses, n: list(obses) + [np.zeros(obses[-1].shape)] * (n - len(obses))
pad_acts = lambda acts, n: list(acts) + [np.zeros(acts[-1].shape)] * (n - len(acts))

def vectorize_demos(demo_data):
  obses = []
  actions = []
  masks = []
  for demo_obses, demo_actions in demo_data:
    for i in range(max(1, len(demo_obses) - history_len + 1)):
      unpadded_obses = demo_obses[i:i+history_len]
      obses.append(pad_obses(unpadded_obses, history_len))
      actions.append(pad_acts(demo_actions[i:i+history_len], history_len))
      masks.append(build_mask(len(unpadded_obses), history_len))
  obses = np.array(obses)
  actions = np.array(actions)
  masks = np.array(masks)
  return obses, actions, masks

def label_actions(rec):
  labels = [None] * len(rec)
  for i, x in enumerate(rec):
    pos = x[0]
    vel = x[1]
    goal = x[-2]
    targ_vel = goal - pos
    action = targ_vel / np.linalg.norm(targ_vel)
    labels[i] = action
  return labels

def featurize_rec(rec):
  return [x[2] for x in rec]

def gen_demos_from_recs(rec_data):
  actions = [label_actions(rec) for rec in rec_data]
  obses = [featurize_rec(rec) for rec in rec_data]
  return list(zip(obses, actions))

def compute_obs_norm(obses):
  mean_obs = np.mean(obses, axis=0)
  std_obs = np.std(obses, axis=0)
  return mean_obs, std_obs

def process_recs(rec_data):
  demo_data = gen_demos_from_recs(rec_data)
  demo_obses, demo_actions, demo_masks = vectorize_demos(demo_data)

  mean_obs, std_obs = compute_obs_norm(demo_obses)

  demo_obses = (demo_obses - mean_obs[np.newaxis, :, :]) / std_obs[np.newaxis, :, :]

  idxes = list(range(demo_obses.shape[0]))
  random.shuffle(idxes)
  n_train_idxes = int(0.9 * len(idxes))
  demo_train_idxes = idxes[:n_train_idxes]
  demo_val_idxes = idxes[n_train_idxes:]

  return {
      'obses': demo_obses,
      'actions': demo_actions,
      'masks': demo_masks,
      'train_idxes': demo_train_idxes,
      'val_idxes': demo_val_idxes,
      'mean_obs': mean_obs,
      'std_obs': std_obs
      }

data_keys = ['obses', 'actions', 'masks']
def sample_batch(data, size=1, idxes_key='train_idxes'):
  idxes = data[idxes_key]
  if size < len(idxes):
    sampled_idxes = random.sample(idxes, size)
  else:
    sampled_idxes = idxes
  batch = {k: data[k][sampled_idxes] for k in data_keys}
  return batch

def build_mlp(
    input_placeholder,
    output_size,
    scope,
    n_layers=1,
    size=256,
    activation=tf.nn.relu,
    output_activation=tf.nn.softmax,
  ):
  outs = [input_placeholder]
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    for _ in range(n_layers):
      outs.append(tf.layers.dense(outs[-1], size, activation=activation))
    outs.append(tf.layers.dense(outs[-1], output_size, activation=output_activation))
  return outs

class RNNDecoderPolicy(object):

  def __init__(self):
    self.hidden_state = None
    self.action = None
    self.obs_feed = np.zeros((1, history_len, n_obs_dim))

  def reset(self):
    self.hidden_state = None
    self.action = None

  def _feed_dict(self, obs):
    self.obs_feed[0, 0, :] = obs
    d = {obs_ph: self.obs_feed}
    if self.hidden_state is not None:
      d[init_state] = self.hidden_state
    return d

  def observe(self, obs):
    with tf.variable_scope(decoder_scope, reuse=tf.AUTO_REUSE):
      self.action, hidden_state = sess.run(
        [rnn_outputs[0], rnn_states[0]], feed_dict=self._feed_dict(obs))
    if history_len > 1:
      self.hidden_state = hidden_state

  def act(self):
    assert self.action.shape[0] == 1
    return self.action[0, :]

class MLPDecoderPolicy(object):

  def __init__(self):
    assert history_len == 1
    self.reset()

  def reset(self):
    self.obs_feed = np.zeros((1, history_len, n_obs_dim))
    self.obs = None

  def _feed_dict(self, obs):
    self.obs_feed[0, 0, :] = obs
    return {obs_ph: self.obs_feed}

  def observe(self, obs):
    self.obs = obs

  def act(self):
    action = sess.run(nn_outputs, feed_dict=self._feed_dict(self.obs))
    return action[0, :]

def compute_batch_loss(batch, update=True):
  feed_dict = {
    obs_ph: batch['obses'],
    act_ph: batch['actions'],
    mask_ph: batch['masks']
  }
  if update:
    [loss_eval, _] = sess.run([loss, update_op], feed_dict=feed_dict)
  else:
    loss_eval = sess.run(loss, feed_dict=feed_dict)
  return loss_eval

with open(os.path.join(curr_dir, 'rec_data.pkl'), 'rb') as f:
  rec_data = pickle.load(f)

def passes_filter(rec):
  conds = []

  # DEBUG
  path = rec[0][-1].split('/')
  conds.append(path[-1] == 'BCI_Fixed')
  #conds.append(path[-4].startswith('201906'))

  conds.append(type(rec[0][-2]) == np.ndarray)
  return all(conds)

rec_data = [rec for rec in rec_data if passes_filter(rec)]

demo_data = process_recs(rec_data)

print('Number of demonstrations: %d' % demo_data['obses'].shape[0])

sess = tf.Session()

decoder_scope = str(uuid.uuid4())
with open(os.path.join(data_dir, 'decoder_scope.pkl'), 'wb') as f:
  pickle.dump(decoder_scope, f, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(data_dir, 'obs_norm.pkl'), 'wb') as f:
  pickle.dump((demo_data['mean_obs'], demo_data['std_obs']), f, pickle.HIGHEST_PROTOCOL)

goals = {tuple(rec[0][-2]) for rec in rec_data}
with open(os.path.join(data_dir, 'goals.pkl'), 'wb') as f:
  pickle.dump(goals, f, pickle.HIGHEST_PROTOCOL)

# setup tensorflow graph
obs_ph = tf.placeholder(tf.float32, [None, history_len, n_obs_dim]) # observations
act_ph = tf.placeholder(tf.float32, [None, history_len, n_act_dim]) # actions
mask_ph = tf.placeholder(tf.float32, [None, history_len]) # masks for RNN training

if history_len > 1:
  with tf.variable_scope(decoder_scope, reuse=tf.AUTO_REUSE):
    weights = {'out': tf.Variable(tf.random_normal([layer_size, n_act_dim]))}
    biases = {'out': tf.Variable(tf.random_normal([n_act_dim]))}

    unstacked_X = tf.unstack(obs_ph, history_len, 1)

    lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=layer_size) for _ in range(n_layers)]
    stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

    init_state = stacked_lstm_cell.zero_state(tf.shape(obs_ph)[0], tf.float32)
    state = init_state
    rnn_outputs = []
    rnn_states = []
    for input_ in unstacked_X:
      output, state = stacked_lstm_cell(input_, state)
      rnn_outputs.append(tf.matmul(output, weights['out']) + biases['out'])
      rnn_states.append(state)

  nn_outputs = tf.concat(rnn_outputs, axis=1)
else:
  nn_outputs = build_mlp(
      tf.squeeze(obs_ph, axis=[1]),
      n_act_dim,
      decoder_scope,
      n_layers=n_layers,
      size=layer_size,
      activation=tf.nn.relu,
      output_activation=None
    )[-1]

reshaped_nn_outputs = tf.reshape(nn_outputs, shape=[tf.shape(obs_ph)[0], history_len, n_act_dim])

# cosine distance loss
reshaped_nn_outputs = tf.math.l2_normalize(reshaped_nn_outputs, axis=2)
cos_dist = 1. - tf.reduce_sum(reshaped_nn_outputs * act_ph, axis=2)
loss = tf.reduce_sum(cos_dist * mask_ph) / tf.reduce_sum(mask_ph)

update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

if history_len > 1:
  nn_decoder_policy = RNNDecoderPolicy()
else:
  nn_decoder_policy = MLPDecoderPolicy()

tf.global_variables_initializer().run(session=sess)

train_data = sample_batch(demo_data, size=len(demo_data['train_idxes']), idxes_key='train_idxes')
val_data = sample_batch(demo_data, size=len(demo_data['val_idxes']), idxes_key='val_idxes')

# DEBUG
nn_activations = build_mlp(
    tf.squeeze(obs_ph, axis=[1]),
    n_act_dim,
    decoder_scope,
    n_layers=n_layers,
    size=layer_size,
    activation=tf.nn.relu,
    output_activation=None
  )[-1]#-2]
def dump_activations():
  activations = []
  for rec in rec_data:
    data = gen_demos_from_recs([rec])
    obses = vectorize_demos(data)[0]
    feed_dict = {obs_ph: obses}
    rec_acts = sess.run(nn_activations, feed_dict=feed_dict)
    activations.append(rec_acts)
  with open(os.path.join(curr_dir, 'activations.pkl'), 'wb') as f:
    pickle.dump(activations, f, pickle.HIGHEST_PROTOCOL)

# main training loop
for t in range(iterations):
  batch = sample_batch(demo_data, size=batch_size, idxes_key='train_idxes')
  compute_batch_loss(batch, update=True)

  if t % print_freq == 0:
    train_loss = compute_batch_loss(train_data, update=False)
    val_loss = compute_batch_loss(val_data, update=False)

    print('%d %d %f %f' % (t, iterations, train_loss, val_loss))

    if t == 0 or val_loss < best_val_loss:
      best_val_loss = val_loss
      save_tf_vars(decoder_scope, os.path.join(data_dir, 'trained_decoder.tf'))
      #dump_activations() # DEBUG

