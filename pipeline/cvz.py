import flair
import lightkurve as lk
import argparse
import h5py as h5
from os.path import isfile

def cvz_pipeline(tic, n_inject, n_repeat, lightkurve_path, out_path, cpu_count, sector=None, sector_ind=0):
    """
    Perform the CVZ pipeline for a given TIC ID and sector.

    Parameters
    ----------
    tic : `str`
        The TIC ID of the target star
    sector : `str`
        The sector number
    sector_ind : `int`
        The index of the sector to use (alternative to `sector`)
    n_inject : `int`
        The number of injections to perform
    n_repeat : `int`
        The number of times to repeat the pipeline
    lightkurve_path : `str`
        The path to a folder for downloading files from lightkurve
    out_path : `str`
        The output path for the pipeline results
    cpu_count : `int`
        The number of CPU cores to use for parallel processing

    Notes
    -----
    This function performs the CVZ pipeline, which includes injecting simulated flares into the target star's light curve,
    running a flare detection algorithm, and saving the results to the specified output path.

    Examples
    --------
    >>> cvz_pipeline(tic="tic272272592", sector='14', n_inject=2000, n_repeat=10, lightkurve_path="/gscratch/scrubbed/tomwagg/",
                     out_path="/gscratch/dirac/flair/cvz/", cpu_count=10)
    """
    lc = None
    file_exists = False
    file_keys = None

    # skip creating the lightcurve and identifying flares if the file already exists
    if isfile(out_path + f"{tic}_sector{sector}.h5"):
        file_exists = True
        with h5.File(out_path + f"{tic}_sector{sector}.h5", "r") as f:
            if "lc" in f:
                lc = lk.LightCurve(time=f["lc/time"][:], flux=f["lc/flux"][:], flux_err=f["lc/flux_err"][:])
                flare_mask = f["lc/flare_mask"][:]

    # if the lightcurve doesn't exist, download it and identify flares
    if lc is None:
        # set the download cache directory
        lk.conf.cache_dir = lightkurve_path

        # download the lightcurve
        lc = flair.lightcurve.get_lightcurve(target=tic, mission='TESS', author='SPOC',
                                            sector=sector, ind=sector_ind)
        
        # setup the CNN and models
        cnn, models = flair.flares.prep_stella('../data')
        
        # predict the flares and create a mask
        avg_pred = flair.flares.get_stella_predictions(cnn=cnn, models=models, lc=lc)
        flare_mask, flare_starts, flare_ends = flair.flares.get_flares(flare_prob=avg_pred, min_flare_points=3,
                                                                    merge_absolute=2, merge_relative=0.2)
        
        # CHECKPOINT 1: save the lightcurve and flare mask
        with h5.File(out_path + f"{tic}_sector{lc.sector}.h5", "w") as f:
            g = f.create_group("lc")
            g.create_dataset("time", data=lc.time)
            g.create_dataset("flux", data=lc.flux)
            g.create_dataset("flux_err", data=lc.flux_err)
            g.create_dataset("flare_mask", data=flare_mask)

    if file_exists and "gp" in file_keys:
        with h5.File(out_path + f"{tic}_sector{lc.sector}.h5", "r") as f:
            mu = f["lc/mu"][:]
            variance = f["lc/variance"][:]
    else:
        # fit the GP to the lightcurve
        opt_gp = flair.gp.fit_GP(lc, flare_mask)
        mu, variance = opt_gp.predict(y=lc.flux.value[~flare_mask], t=lc.time.value, return_var=True)

        # CHECKPOINT 2: save the GP mean and variance
        with h5.File(out_path + f"{tic}_sector{lc.sector}.h5", "a") as f:
            g = f["lc"]
            g.create_dataset("mu", data=mu)
            g.create_dataset("variance", data=variance)


# setup argparse
def main():
    parser = argparse.ArgumentParser(description='Run CVZ pipeline on a specific sector')
    parser.add_argument('-t', '--tic', default="", type=str,
                        help='TIC ID of the star')
    parser.add_argument('-s', '--sector', default="", type=str,
                        help='ID of the sector to use (e.g. "14")')
    parser.add_argument('-i', '--sector-ind', default="", type=int,
                        help='Alternative to sector, the index of the sector to use (e.g. 2, meaning 2nd sector)')
    parser.add_argument('-ni', '--ninject', default=2000, type=int,
                        help='Number of flares to inject')
    parser.add_argument('-nr', '--nrepeat', default=10, type=int,
                        help='Number of times to repeat each injected flare')
    parser.add_argument('-l', '--lightkurve-path', default="/gscratch/scrubbed/tomwagg/", type=str,
                        help='Path to use for downloading lightkurve files')
    parser.add_argument('-o', '--out-path', default="/gscratch/dirac/flair/cvz/", type=str,
                        help='Path to use for output files')
    parser.add_argument('-c', '--cpu-count', default=10, type=int,
                        help='How many CPUs to use for the injection and recovery tests')
    args = parser.parse_args()

    # run the pipeline
    cvz_pipeline(tic=args.tic, sector=args.sector, sector_ind=args.sector_ind, n_inject=args.ninject,
                 n_repeat=args.nrepeat, lightkurve_path=args.lightkurve_path, out_path=args.out_path,
                 cpu_count=args.cpu_count)


if __name__ == "__main__":
    main()