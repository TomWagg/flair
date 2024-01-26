import stella
import numpy as np


def prep_stella(out_path='data'):
    # set up stellar CNN
    cnn = stella.ConvNN(output_dir=out_path)

    # download the stellar models
    ds = stella.download_nn_set.DownloadSets()
    ds.download_models()

    return cnn, ds.models


def get_stella_predictions(cnn, models, lc):
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


def get_flares(flare_prob, threshold=0.3, min_flare_points=3):
    """Get a mask for timesteps that are part of a flare and indices of flare starts and ends.

    Parameters
    ----------
    flare_prob : :class:`numpy.ndarray`
        Probability of a flare at each timestep
    threshold : `float`, optional
        Threshold probability, by default 0.3
    min_flare_points : `int`, optional
        Minimum number of contiguous timesteps necessary for a flare, by default 3

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
    all_flares = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

    # remove blocks with less than min_flare_points
    long_enough_flares = [f for f in all_flares if len(f) >= min_flare_points]

    # get indices of flare starts and ends
    flare_starts = [f[0] for f in long_enough_flares]
    flare_ends = [f[-1] for f in long_enough_flares]

    # create a mask for timesteps that are part of a flare
    flare_mask = np.zeros(len(flare_prob), dtype=bool)
    flare_inds = np.concatenate(long_enough_flares)
    flare_mask[flare_inds] = True

    return flare_mask, flare_starts, flare_ends
