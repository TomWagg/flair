import stella
import numpy as np


def prep_stella(out_path='data'):
    """Set up the Stella ConvNN object and download the stellar models.

    Parameters
    ----------
    out_path : `str`, optional
        Where to store the data, by default 'data'

    Returns
    -------
    cnn : :class:`stella.ConvNN`
        The Stella ConvNN object
    models : :class:`list`
        List of trained stellar models
    """
    # set up stellar CNN
    cnn = stella.ConvNN(output_dir=out_path)

    # download the stellar models
    ds = stella.download_nn_set.DownloadSets()
    ds.download_models()

    return cnn, ds.models


def get_stella_predictions(cnn, models, lc):
    """Get the median flare probability across all models for each timestep in a lightcurve using Stella.

    Parameters
    ----------
    cnn : :class:`stella.ConvNN`
        The Stella ConvNN object
    models : :class:`list`
        List of trained stellar models
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to predict on

    Returns
    -------
    avg_pred : :class:`numpy.ndarray`
        Median flare probability across all models for each timestep in the lightcurve
    """
    # predict the flare probability for each model
    preds = np.zeros((len(models), len(lc)))
    for j, model in enumerate(models):
        cnn.predict(modelname=model,
                    times=lc.time.value,
                    fluxes=lc.flux.value,
                    errs=lc.flux_err.value)
        preds[j] = cnn.predictions[0]

    # return the median probability across all models, ignoring NaNs
    return np.nanmedian(preds, axis=0)


def get_flares(flare_prob, threshold=0.3, min_flare_points=3, merge_absolute=2, merge_relative=0.2):
    """Get a mask for timesteps that are part of a flare and indices of flare starts and ends.

    Parameters
    ----------
    flare_prob : :class:`numpy.ndarray`
        Probability of a flare at each timestep
    threshold : `float`, optional
        Threshold probability, by default 0.3
    min_flare_points : `int`, optional
        Minimum number of contiguous timesteps necessary for a flare, by default 3
    merge_absolute : `int`, optional
        Merge flares that are closer than this number of timesteps, by default 2
    merge_relative : `float`, optional
        Merge flares that are closer than this fraction of the flare duration, by default 0.2

    Returns
    -------
    flare_mask : :class:`numpy.ndarray`
        Boolean array with True for timesteps that are part of a flare
    flare_starts : :class:`numpy.ndarray`
        Indices of flare starts
    flare_ends : :class:`numpy.ndarray`
        Indices of flare ends
    """
    # get indices above threshold
    idx = np.where(flare_prob > threshold)[0]

    # split into contiguous blocks of indices
    flares = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

    # merge blocks that are closer than merge_absolute or merge_relative
    merged_flares = [flares[0]]
    for i in range(1, len(flares)):
        # define some variables for convenience
        current_start, most_recent_end = flares[i][0], merged_flares[-1][-1]
        most_recent_duration = most_recent_end - merged_flares[-1][0]
        gap_size = current_start - most_recent_end

        # merge if gap is small enough (either in absolute or relative terms)
        if gap_size < merge_absolute or gap_size < merge_relative * most_recent_duration:
            merged_flares[-1] = np.arange(merged_flares[-1][0], flares[i][-1] + 1, dtype=int)
        # otherwise add the new flare to the list
        else:
            merged_flares.append(flares[i])

    # remove blocks with less than min_flare_points
    long_enough_flares = [f for f in merged_flares if len(f) >= min_flare_points]

    # get indices of flare starts and ends
    flare_starts = np.array([f[0] for f in long_enough_flares])
    flare_ends = np.array([f[-1] for f in long_enough_flares])

    # create a mask for timesteps that are part of a flare
    flare_mask = np.zeros(len(flare_prob), dtype=bool)
    flare_inds = np.concatenate(long_enough_flares)
    flare_mask[flare_inds] = True

    return flare_mask, flare_starts, flare_ends
