import gym
import sys
import CursorControl
import matplotlib.pyplot as plt

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter

env = gym.make('cursorcontrol-v1')
log_path = "%s" % sys.argv[1]

model = SAC(MlpPolicy, env, verbose=1)
model.load(log_path)
time_steps = int(1e6)

results_plotter.plot_results([log_path], time_steps, results_plotter.X_TIMESTEPS, "SAC CursorControl")
plt.show()

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()