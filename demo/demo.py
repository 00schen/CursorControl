import gym
import CursorControl
import matplotlib.pyplot as plt

from stable_baselines import SAC
from stable_baselines import results_plotter

from proxy_draw_curve import make_plot_curves

log_path = '../logs/sac_pred_goal2'
env_name = 'goalcontrol-v2'

rollout = 3
oracle_dim = 16
noise,gamma = .05,.9
penalty = 100
time_steps = int(5e5)

results_plotter.plot_curves = make_plot_curves([100])
results_plotter.plot_results([log_path], time_steps, results_plotter.X_EPISODES, "SAC GoalControl")
plt.show()

for r in range(time_steps-int(5e4),time_steps+int(5e4),int(5e4)):
  real_log_path = log_path+"/rl_model_%d_steps"%r
  model = SAC.load(real_log_path)

  params = {'oracle_noise':noise, 'oracle_dim':oracle_dim, 'rollout':rollout, 'penalty':penalty}
  env = gym.make(env_name,**params)

  for _ in range(5):
    obs = env.reset()
    for i in range(100):
      action, _states = model.predict(obs)
      obs, rewards, done, info = env.step(action)s
      env.render(str(r))
      if done:
        break

env.close()