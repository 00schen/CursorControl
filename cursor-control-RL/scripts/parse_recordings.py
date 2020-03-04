from __future__ import division
import pickle
import os

import numpy as np

import sys
sys.path.append(os.path.join('/home', 'sid', 'BCI-Utilities'))
import bci

params = bci.task_management.get_params()
rec_dir = os.path.join('/home', 'sid', 'cursor-control-RL', 'data', 'cursor-control', 'Bravo1_PythonData')
bci_dim = 128*7
screen_size = np.array([950, 530])
normalize_pos = lambda p: (p + screen_size) / (2 * screen_size)

def process_recording(raw_rec, rec_path):
  goal = normalize_pos(raw_rec['TargetPosition'])

  ix_trial_bin = bci.task_management.get_ix_trial(raw_rec)
  contexts = raw_rec['CursorState'].T
  contexts = contexts[ix_trial_bin, :]

  feat_neural, _ = bci.signal_processing.process_bin_signals(
    raw_rec['BroadbandData']['signals'], params['neural_features'],
    params['signal_processing'], bins=raw_rec['BroadbandData']['bin_lens'],
    use_bin=ix_trial_bin, axis=0)

  bci_feats = np.zeros((len(feat_neural), bci_dim))
  for t, feat_neural_at_t in enumerate(feat_neural):
    i = 0
    for band_feats in feat_neural_at_t.values():
      for vec in band_feats.values():
        bci_feats[t, i:i+vec.size] = vec
        i += vec.size
    assert i == bci_dim

  assert contexts.shape[0] == bci_feats.shape[0]

  contexts[:, :2] = normalize_pos(contexts[:, :2])

  T = contexts.shape[0]
  data = [None] * T
  for t in range(T):
    pos = contexts[t, :2]
    vel = pos - contexts[max(t-1,0), :2]
    data[t] = (pos, vel, bci_feats[t, :], goal, rec_path)

  return data

def save_rec_data(rec_data, save_path='rec_data.pkl'):
  with open(save_path, 'wb') as f:
    pickle.dump(rec_data, f, pickle.HIGHEST_PROTOCOL)

save_freq = 500

n_examples = 0
skipped = 0
rec_data = []
print('n_recs\tn_examples')
for root, dirs, files in os.walk(rec_dir):
  for rec_file in files:
    if rec_file.startswith('Data') and rec_file.endswith('.pkl'):
      with open(os.path.join(root, rec_file), 'rb') as f:
        try:
          raw_rec = pickle.load(f)
          rec = process_recording(raw_rec, root)
          rec_data.append(rec)
          n_examples += len(rec)
          print('%d\t%d' % (len(rec_data), n_examples))
          if len(rec_data) % save_freq == 5:
            save_rec_data(rec_data)
        except:
          skipped += 1

save_rec_data(rec_data)

print('Could not parse %d recordings' % skipped)

