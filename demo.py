import gym
import sys
import CursorControl
import matplotlib.pyplot as plt

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines.results_plotter import ts2xy, load_results, plot_curves, X_EPISODES

def plot_results(dirs, num_timesteps, xaxis, task_name):
    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name)
    plt.legend(zip(*xy_list),[s.replace("sac_best_","") for s in dirs])

env = gym.make('cursorcontrol-v1')
log_path = "%s" % sys.argv[1]
best_model_save_path = "%s" % sys.argv[2]

model = SAC(MlpPolicy, env, verbose=1)
model.load(best_model_save_path)
time_steps = int(1e5)

plot_results([log_path], time_steps, X_EPISODES, "SAC CursorControl")
plt.show()

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()