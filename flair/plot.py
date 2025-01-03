import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import flair


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
        returned = func(*args, **kwargs)
        if 'show' in kwargs and kwargs['show'] or 'show' not in kwargs:
            plt.show()
        if returned is not None:
            return (*returned, fig, ax) if isinstance(returned, tuple) else (returned, fig, ax)
        else:
            return fig, ax
    return wrapper


@plot_decorator
def plot_lc_with_probs(lc, avg_pred, fig=None, ax=None, show=True):
    """Plot a lightcurve with the flare probability as a colorbar.

    Parameters
    ----------
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to plot
    avg_pred : :class:`numpy.ndarray`
        Flare probability for each timestep in the lightcurve
    fig : :class:`matplotlib.figure.Figure`, optional
        Figure to plot on, by default None (new figure created)
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes to plot on, by default None (new axes created)
    show : `bool`, optional
        Whether to show the plot, by default True

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Figure of plot
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes of plot
    """
    scatter = ax.scatter(lc.time.value, lc.flux.value, c=avg_pred,
                         vmin=0, vmax=1, label=f'Sector {lc.sector}')
    fig.colorbar(scatter, label="Flare Probability")

    ax.set(xlabel='Time [BJD-2457000]', ylabel='Normalized Flux')
    ax.legend()

@plot_decorator
def plot_lc_with_flares(lc, flare_mask, base_col="black", flare_col="tab:orange", 
                        fig=None, ax=None, show=True, **kwargs):
    """Plot a lightcurve with the flare probability as a colorbar.

    Parameters
    ----------
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to plot
    flare_mask : :class:`numpy.ndarray`
        Boolean array with True for timesteps that are part of a flare
    fig : :class:`matplotlib.figure.Figure`, optional
        Figure to plot on, by default None (new figure created)
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes to plot on, by default None (new axes created)
    show : `bool`, optional
        Whether to show the plot, by default True

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Figure of plot
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes of plot
    """
    colours = np.where(flare_mask, flare_col, base_col)
    ax.scatter(lc.time.value, lc.flux.value / np.median(lc.flux.value),
               c=colours, label=kwargs.pop("label", f'Sector {lc.sector}'), s=kwargs.pop("s", 10), **kwargs)
    ax.set(xlabel='Time [BJD-2457000]', ylabel='Normalized Flux')
    ax.legend()


@plot_decorator
def plot_lc_and_gp(lc, flare_mask, flare_starts=None, flare_ends=None,
                   mu=None, time_lims=None, highlight_flares=True,
                   fig=None, ax=None, show=True):
    """Plot a lightcurve with flares masked and the GP prediction.

    Parameters
    ----------
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to plot
    flare_mask : :class:`numpy.ndarray`
        Boolean array with True for timesteps that are part of a flare
    flare_starts : :class:`numpy.ndarray`
        Indices of flare starts
    flare_ends : :class:`numpy.ndarray`
        Indices of flare ends
    mu : :class:`numpy.ndarray`, optional
        GP mean, by default None (calculated from gp.predict)
    time_lims : `tuple`, optional
        Time limits for the plot, by default None
    highlight_flares : `bool`, optional
        Whether to highlight flares on the plot, by default True
    fig : :class:`matplotlib.figure.Figure`, optional
        Figure to plot on, by default None (new figure created)
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes to plot on, by default None (new axes created)
    show : `bool`, optional
        Whether to show the plot, by default True

    Returns
    -------
    mu : :class:`numpy.ndarray`
        GP mean
    fig : :class:`matplotlib.figure.Figure`
        Figure of plot
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes of plot
    """
    # create a mask based on the time limits
    if time_lims is not None:
        time_mask = (lc.time.value > time_lims[0]) & (lc.time.value < time_lims[1])
    else:
        time_mask = np.ones_like(lc.time.value).astype(bool)
        
    # calculate the GP prediction if not provided
    if mu is None:
        mu= flair.gp.fit_gp_tiny(lc, flare_mask)

    # plot the lightcurve with flares masked
    ax.plot(lc.time.value[~flare_mask & time_mask], lc.flux.value[~flare_mask & time_mask],
            label="Data (flares masked)", alpha=1, color="C0")

    # plot the GP prediction
    ax.plot(lc.time.value, mu, label="GP Prediction", color="C1")

    # add a legend
    ax.legend(fontsize=0.7 * fs)

    # highlight flares if desired
    if highlight_flares:
        if flare_starts is None or flare_ends is None:
            raise ValueError("Flare starts and ends must be provided to highlight flares.")
        for s, e in zip(flare_starts, flare_ends):
            start, end = lc.time.value[s], lc.time.value[e]
            if time_lims is None or (start > time_lims[0] and end < time_lims[1]):
                ax.axvspan(start, end, alpha=0.15, color="grey")

    # set the axis labels
    ax.set(xlabel="Time [BJD - 2457000, days]", ylabel="Flux [e-/s]")

    return mu, fig
