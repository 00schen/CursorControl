import gym
import sys
import CursorControl
import matplotlib.pyplot as plt

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter

from proxy_draw_curve import make_plot_curves

log_path = "../logs/sac_penalty"

time_steps = int(2e4)
rollout = 3
noise,gamma = .0,.9
penalties = [100]


results_plotter.plot_curves = make_plot_curves(penalties)
results_plotter.plot_results([log_path+"_%d"%r for r in penalties], time_steps, results_plotter.X_EPISODES, "SAC CursorControl")
plt.show()

for r in penalties:
  # real_log_path = log_path+"_%d"%r
  # model = SAC.load(real_log_path+'/best_model')

  params = {'oracle_noise':noise, 'rollout':rollout}
  env = gym.make('cursorcontrol-v2',**params)
  print(env.ORACLE_NOISE)

  # obs = env.reset()
  # for i in range(100):
  #   action, _states = model.predict(obs)
  #   obs, rewards, done, info = env.step(action)
  #   env.render(str(r))
  #   if done:
  #     break

env.close()