import matplotlib.pyplot as plt
import matplotlib as mpl


plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


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
    scatter = ax.scatter(lc.time.value, lc.flux.value, c=avg_pred,
                         vmin=0, vmax=1, label=f'Sector {lc.sector}')
    fig.colorbar(scatter, label="Flare Probability")

    ax.set(xlabel='Time [BJD-2457000]', ylabel='Normalized Flux')
    ax.legend()


@plot_decorator
def plot_lc_and_gp(lc, flare_mask, gp=None, mu=None, variance=None, fig=None, ax=None, show=True):

    if mu is None or variance is None:
        mu, variance = gp.predict(y=lc.flux.value, t=x, return_var=True)

    ax.plot(lc.time.value, lc.flux.value, label='Lightcurve')
    ax.plot(lc.time.value[flare_mask], lc.flux.value[flare_mask], 'o', label='Flares')
    ax.plot(lc.time.value, gp.predict(lc.flux.value, return_cov=False), label='GP')

    ax.set(xlabel='Time [BJD-2457000]', ylabel='Normalized Flux')
    ax.legend()