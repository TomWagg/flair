from . import lupita
from . import flares
import numpy as np
from astropy.table import Table
from multiprocessing import Pool

__all__ = ['inject_flare', 'is_recovered', 'injection_test', 'each_flare', 'amplitude_to_energy']

# fits from Althukair+23 (https://arxiv.org/abs/2212.10224)
_amp_E_lines = {'G': np.poly1d([ 1.18996884, 37.27283556]),
                'K': np.poly1d([ 1.18075204, 36.61235441]),
                'M': np.poly1d([ 1.75999376, 36.05875657])}
_amp_E_scatter = {'G': 0.1754525572603882, 'K': 0.21504357742404395, 'M': 0.27823866674159486}


def inject_flare(time, flux, amp, fwhm, insert_timestep, flare_mask):
    """Inject a flare into a lightcurve.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Time values for the lightcurve
    flux : :class:`numpy.ndarray`
        Flux values for the lightcurve
    amp : `float`
        Amplitude of the flare
    fwhm : `float`
        Full Width at Half Maximum of the flare
    insert_timestep : `int`
        Index of the timestep to insert the flare at
    flare_mask : :class:`numpy.ndarray`
        Boolean array with True for timesteps that are part of a flare

    Returns
    -------
    adjusted_lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve with the flare injected
    """
    model_flux = lupita.flare_model(time, time[insert_timestep], fwhm, amp * np.median(flux[~flare_mask]))
    return flux + np.array(np.nan_to_num(model_flux, nan=0.0))

def is_recovered(cnn, models, time, flux, flux_err, timestep, threshold=0.3, min_flare_points=3):
    """Check if an injected flare is recovered at a given timestep in a lightcurve.

    Parameters
    ----------
    cnn : :class:`stella.ConvNN`
        The Stella ConvNN object
    models : :class:`list`
        List of trained stellar models
    time : :class:`numpy.ndarray`
        Time values for the lightcurve
    flux : :class:`numpy.ndarray`
        Flux values for the lightcurve
    flux_err : :class:`numpy.ndarray`
        Flux error values for the lightcurve
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
    avg_pred = flares.get_stella_predictions(cnn=cnn, models=models, time=time, flux=flux, flux_err=flux_err)
    offset = (min_flare_points - 1) // 2
    return np.all(avg_pred[timestep - offset:timestep + offset] > threshold)

def injection_test(time, flux, flux_err, cnn, models, flare_mask, amp, fwhm, insertion_point):
    """Test the recovery of an injected flare.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Time values for the lightcurve
    flux : :class:`numpy.ndarray`
        Flux values for the lightcurve
    flux_err : :class:`numpy.ndarray`
        Flux error values for the lightcurve
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

    adjusted_flux = inject_flare(time=time, flux=flux, amp=amp, fwhm=fwhm,
                                 insert_timestep=insertion_point, flare_mask=flare_mask)
    return is_recovered(cnn=cnn, models=models, time=time, flux=adjusted_flux, flux_err=flux_err,
                        timestep=insertion_point)


def evaluate_completeness(lc, flare_mask, cnn=None, models=None,
                          n_end_avoid=5, n_inject=50, n_repeat=10, processes=1):
    """Evaluate the completeness of stella for a given lightcurve.
    
    Inject flares and test their recovery. Return details of the injected flares and the recovery rate.

    Parameters
    ----------
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to test
    flare_mask : :class:`numpy.ndarray`
        Boolean array with True for timesteps that are part of a flare
    cnn : :class:`stella.ConvNN`
        The Stella ConvNN object, by default None (created if not provided)
    models : :class:`list`
        List of trained stellar models, by default None (created if not provided)
    n_end_avoid : `int`, optional
        Number of timesteps to avoid at the start and end of the lightcurve, by default 5
    n_inject : `int`, optional
        Number of flares to inject, by default 50
    n_repeat : `int`, optional
        Number of times to repeat the test on each flare, by default 10

    Returns
    -------
    recovered : :class:`numpy.ndarray`, shape = (n_inject, n_repeat)
        Boolean array with True for each flare that was recovered
    """
    # flare amplitudes are set by the typical uncertainty in the lightcurve
    norm_median_error = np.median(lc.flux_err) / np.median(lc.flux)
    amps = np.logspace(np.log10(norm_median_error), np.log10(10 * norm_median_error), n_inject)

    # convert the amplitudes to energies
    energies = amplitude_to_energy(amps, "G")

    # calculate the FWHM of the flares based on Lupita's model
    fwhms = (energies / 2.0487) * amps

    # draw random injection times
    all_inds = np.arange(len(lc))
    not_flare_inds = all_inds[~flare_mask & (all_inds > n_end_avoid) & (all_inds < len(lc) - n_end_avoid)]
    insert_points = np.random.choice(not_flare_inds, size=(n_inject, n_repeat))

    recovered = np.zeros((n_inject, n_repeat), dtype=bool)

    # # get the time, flux, and flux_err from the lightcurve in simple ndarrays (pools are picky)
    time, flux, flux_err = np.array(lc.time.value), np.array(lc.flux.value), np.array(lc.flux_err.value)

    # # create a generator to pass to the parallel processing function
    def args(amps, fwhms, insert_points):
        for amp, fwhm, ip in zip(amps, fwhms, insert_points):
            yield time, flux, flux_err, cnn, models, flare_mask, amp, fwhm, ip

    # if the user wants to use parallel processing, do so
    if processes > 1:
        with Pool(processes) as pool:
            for i in range(n_repeat):
                recovered[:, i] = list(pool.starmap(injection_test, args(amps, fwhms, insert_points[:, i])))
    # otherwise, just loop through the flares one at a time
    else:
        for i in range(n_repeat):
            recovered[:, i] = [injection_test(*arg) for arg in args(amps, fwhms, insert_points[:, i])]

    return recovered


def amplitude_to_energy(amp, stellar_class):
    """Convert a flare amplitude to an energy.
    
    Using the fits from Althukair+23 (https://arxiv.org/abs/2212.10224) convert a flare amplitude to an energy
    accounting for the scatter in the relationship and difference for different stellar classes.

    Parameters
    ----------
    amp : :class:`numpy.ndarray` or `float`
        Flare amplitude
    stellar_class : `str`
        Stellar class

    Returns
    -------
    E : :class:`numpy.ndarray`
        Flare energy in erg
    """
    n_flares = len(amp) if isinstance(amp, np.ndarray) else 1
    scatter = np.random.normal(0, _amp_E_scatter[stellar_class], n_flares)
    return 10**(_amp_E_lines[stellar_class](np.log10(amp)) + scatter)