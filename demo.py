import gym
import CursorControl
import matplotlib.pyplot as plt

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter

from proxy_draw_curve import make_plot_curves

log_path = "logs/sac_pred_goal/rl_model_"

rollout = 3
noise,gamma = .05,.9
penalty = 100


for r in range(int(4.5e5),int(5.5e5),int(5e4)):
  real_log_path = log_path+"%d_steps"%r
  model = SAC.load(real_log_path)

  params = {'oracle_noise':noise, 'rollout':rollout, 'penalty':penalty}
  env = gym.make('cursorcontrol-v1',**params)

  for _ in range(5):
    obs = env.reset()
    for i in range(100):
      action, _states = model.predict(obs)
      obs, rewards, done, info = env.step(action)
      env.render(str(r))
      if done:
        break

env.close()