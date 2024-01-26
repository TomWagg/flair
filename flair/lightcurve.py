import lightkurve as lk
import numpy as np

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

    # @TOBIN: Do we need this?
    search.table["dataURL"] = search.table["dataURI"]

    lcs = [search[i].download().PDCSAP_FLUX for i in range(len(search))]
    return [lc[~np.isnan(lc.flux.value)] for lc in lcs]
