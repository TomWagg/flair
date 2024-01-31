from . import lupita
from . import flares
import numpy as np
from astropy.table import Table
from multiprocessing import Pool

def inject_flare(lc, amp, fwhm, insert_timestep):
    """Inject a flare into a lightcurve.

    Parameters
    ----------
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to inject the flare into
    amp : `float`
        Amplitude of the flare
    fwhm : `float`
        Full Width at Half Maximum of the flare
    insert_timestep : `int`
        Index of the timestep to insert the flare at

    Returns
    -------
    adjusted_lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve with the flare injected
    """
    # TODO: should the median be taken over the *entire* lightcurve or just the parts without flares?
    model_flux = lupita.flare_model(lc.time.value, lc.time.value[insert_timestep],
                                    fwhm, amp * np.median(lc.flux.value))
    adjusted_lc = lc.copy()
    adjusted_lc["flux"] = (lc.flux.value + np.nan_to_num(model_flux, nan=0.0)) * lc.flux.unit
    return adjusted_lc

def is_recovered(cnn, models, lc, timestep, threshold=0.3, min_flare_points=3):
    """Check if an injected flare is recovered at a given timestep in a lightcurve.

    Parameters
    ----------
    cnn : :class:`stella.ConvNN`
        The Stella ConvNN object
    models : :class:`list`
        List of trained stellar models
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to check for the flare in
    timestep : `int`
        Timestep to check for the flare at
    threshold : `float`, optional
        Threshold probability required, by default 0.3
    min_flare_points : `int`, optional
        Minimum number of consecutive flaring timesteps required (centred on `timestep`), by default 3

    Returns
    -------
    recovered : `bool`
        Whether the flare was recovered at the given timestep
    """
    avg_pred = flares.get_stella_predictions(cnn, models, lc)
    offset = (min_flare_points - 1) // 2
    return np.all(avg_pred[timestep - offset:timestep + offset] > threshold)

def injection_test(lc, cnn, models, flare_mask, amp, fwhm, n_end_avoid=5, **kwargs):
    """Test the recovery of an injected flare.

    Parameters
    ----------
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to test
    cnn : :class:`stella.ConvNN`
        The Stella ConvNN object
    models : :class:`list`
        List of trained stellar models
    flare_mask : :class:`numpy.ndarray`
        Boolean array with True for timesteps that are part of a flare
    amp : `float`
        Amplitude of the flare
    fwhm : `float`
        Full Width at Half Maximum of the flare
    n_end_avoid : `int`, optional
        Number of timesteps to avoid at the start and end of the lightcurve, by default 5

    Returns
    -------
    recovered : `bool`
        Whether the injected flare was recovered
    """
    all_inds = np.arange(len(lc))
    not_flare_inds = all_inds[~flare_mask & (all_inds > n_end_avoid) & (all_inds < len(lc) - n_end_avoid)]
    rand_insertion_point = np.random.choice(not_flare_inds)

    adjusted_lc = inject_flare(lc, amp, fwhm, rand_insertion_point)
    return is_recovered(cnn, models, adjusted_lc, rand_insertion_point, **kwargs)


def each_flare(lc, cnn, models, flare_mask, flare_table_path, n_end_avoid=5, n_repeat=10, processes=1):
    """Test the recovery of each flare in a lightcurve.

    Parameters
    ----------
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to test
    cnn : :class:`stella.ConvNN`
        The Stella ConvNN object
    models : :class:`list`
        List of trained stellar models
    flare_mask : :class:`numpy.ndarray`
        Boolean array with True for timesteps that are part of a flare
    flare_table_path : `str`
        Path to the flare table
    n_end_avoid : `int`, optional
        Number of timesteps to avoid at the start and end of the lightcurve, by default 5
    n_repeat : `int`, optional
        Number of times to repeat the test, by default 10

    Returns
    -------
    recovered : :class:`numpy.ndarray`, shape = (n_flares, n_repeat)
        Boolean array with True for each flare that was recovered
    """
    # read in the synthetic flares and create an array to store the results
    synthetic_flares = Table.read(flare_table_path, format='mrt')
    recovered = np.zeros((len(synthetic_flares), n_repeat), dtype=bool)
    amps, fwhms = synthetic_flares["Amp"].value, synthetic_flares["FWHM"].value

    # create a generator to pass to the parallel processing function
    def args(amps, fwhms):
        for amp, fwhm in zip(amps, fwhms):
            yield lc, cnn, models, flare_mask, amp, fwhm, n_end_avoid

    # if the user wants to use parallel processing, do so
    if processes > 1:
        with Pool(processes) as pool:
            for i in range(n_repeat):
                recovered[:, i] = list(pool.starmap(injection_test, args(amps, fwhms)))
    # otherwise, just loop through the flares one at a time
    else:
        for i in range(n_repeat):
            recovered[:, i] = [injection_test(*arg) for arg in args(amps, fwhms)]

    return recovered
