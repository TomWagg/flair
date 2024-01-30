import lightkurve as lk
import numpy as np

__all__ = ["get_all_lightcurves"]

def _search_obj_to_lc(s):
    """Download a lightcurve from a search result and reduce to time, flux, and flux_err.

    Parameters
    ----------
    s : :class:`lightkurve.search.SearchResult`
        Search result object

    Returns
    -------
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve with time, flux, and flux_err
    """
    # download the full lightcurve
    lc = s.download()

    # reduce to just time, flux, and flux_err (using PDCSAP flux)
    lc = lc[["time", "pdcsap_flux", "pdcsap_flux_err"]]
    lc["flux"] = lc["pdcsap_flux"]
    lc["flux_err"] = lc["pdcsap_flux_err"]
    lc.remove_columns(["pdcsap_flux", "pdcsap_flux_err"])

    # return lightcurve with NaNs removed
    return lc[~np.isnan(lc.flux)]

def get_all_lightcurves(**kwargs):
    """Get all lightcurves matching the given search criteria.

    Returns
    -------
    lcs : :class:`list`
        List of :class:`lightkurve.lightcurve.TessLightCurve` objects

    See Also
    --------
    :func:`lightkurve.search_lightcurve`
    """
    search = lk.search_lightcurve(**kwargs)
    search.table["dataURL"] = search.table["dataURI"]
    return [_search_obj_to_lc(s) for s in search]
