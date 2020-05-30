import gym
import sys
import CursorControl
import matplotlib.pyplot as plt

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter

from proxy_draw_curve import make_plot_curves

log_path = "logs/sac_noise"

time_steps = int(1.5e5)
noises = [0,5e-2,1e-1]


results_plotter.plot_curves = make_plot_curves(noises)
results_plotter.plot_results([log_path+"_%f"%noise for noise in noises], time_steps, results_plotter.X_EPISODES, "SAC CursorControl")
plt.show()

for noise in noises:
  real_log_path = log_path+"_%f"%noise
  model = SAC.load(real_log_path+'/best_model')
  env = gym.make('cursorcontrol-v1')
  env.set_oracle_noise(noise)
  obs = env.reset()
  for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
      break

env.close()