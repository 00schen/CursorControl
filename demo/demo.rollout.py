import gym
import sys
import CursorControl
import matplotlib.pyplot as plt

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter

from proxy_draw_curve import make_plot_curves

log_path = "logs/sac_rollout_no"

time_steps = int(2e4)
rollouts = [3,5,7,10]
noise,gamma = .0,.9


results_plotter.plot_curves = make_plot_curves(rollouts)
results_plotter.plot_results([log_path+"_%f"%r for r in rollouts], time_steps, results_plotter.X_EPISODES, "SAC CursorControl")
plt.show()

for r in rollouts:
  real_log_path = log_path+"_%f"%r
  model = SAC.load(real_log_path+'/best_model')
  env = gym.make('cursorcontrol-v2')
  env.set_oracle_noise(noise)
  env.set_rollout(r)
  obs = env.reset()
  for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
      break

env.close()