from stable_baselines.results_plotter import *

import matplotlib.pyplot as plt

from stable_baselines.bench.monitor import load_results

plt.rcParams['svg.fonttype'] = 'none'

def make_plot_curves(labels):

    def plot_curves(xy_list, xaxis, title):
        """
        plot the curves
        :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
        :param xaxis: (str) the axis for the x and y output
            (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
        :param title: (str) the title of the plot
        """

        plt.figure(figsize=(8, 2))
        maxx = max(xy[0][-1] for xy in xy_list)
        minx = 0
        artists = []
        for (i, (x, y)) in enumerate(xy_list):
            color = COLORS[i]
            artists.append(plt.scatter(x, y, s=2)[0])
            # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
            if x.shape[0] >= EPISODES_WINDOW:
                # Compute and plot rolling mean with window of size EPISODE_WINDOW
                x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
                plt.plot(x, y_mean, color=color)
        plt.xlim(minx, maxx)
        plt.title(title)
        plt.xlabel(xaxis)
        plt.ylabel("Episode Rewards")
        plt.tight_layout()
        plt.legend(artists, labels)
    
    return plot_curves
