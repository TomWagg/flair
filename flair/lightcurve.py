import lightkurve as lk
import numpy as np

def get_all_lightcurves(**kwargs):
    search = lk.search_lightcurve(**kwargs)
    search.table["dataURL"] = search.table["dataURI"]

    lcs = [search[i].download().PDCSAP_FLUX for i in range(len(search))]
    return [lc[~np.isnan(lc.flux.value)] for lc in lcs]
