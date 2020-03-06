import gym
import sys
import CursorControl
import matplotlib.pyplot as plt

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter

env = gym.make('cursorcontrol-v1')
log_path = "%s" % sys.argv[1]
best_model_save_path = "%s" % sys.argv[2]

model = SAC(MlpPolicy, env, verbose=1)
model.load(best_model_save_path)
time_steps = int(1e5)

results_plotter.plot_results([log_path], time_steps, results_plotter.X_EPISODES, "SAC CursorControl")
plt.show()

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()