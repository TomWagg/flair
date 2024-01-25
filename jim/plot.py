import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_decorator(func):
    """A decorator for plotting functions that creates a figure and axes if they are not provided.
    Also shows the plot if the show keyword is set to True or not provided.

    Parameters
    ----------
    func : `function`
        The plotting function to decorate
    """
    def wrapper(*args, **kwargs):
        if 'fig' not in kwargs or ax not in 'kwargs':
            fig, ax = plt.subplots()
            kwargs['fig'] = fig
            kwargs['ax'] = ax
        func(*args, **kwargs)
        if 'show' in kwargs and kwargs['show'] or 'show' not in kwargs:
            plt.show()
        return fig, ax
    return wrapper


@plot_decorator
def plot_lc_with_stella(lc, avg_pred, fig=None, ax=None, show=True):
    scatter = ax.scatter(lc.time, lc.flux, c=avg_pred,
                         vmin=0, vmax=1, label=f'Sector {lc.sector}')
    fig.colorbar(scatter, label="Flare Probability")

    ax.set(xlabel='Time [BJD-2457000]', ylabel='Normalized Flux')
    ax.legend()
