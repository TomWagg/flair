
import argparse
import h5py as h5
from os.path import isfile, join
import logging
import warnings

import numpy as np
import lightkurve as lk

import sys
sys.path.append("../")

import flair

def cvz_pipeline(tic, n_inject, n_repeat, cache_path, out_path, cpu_count, sector_ind=0, log_level="NOTSET"):
    """
    Perform the CVZ pipeline for a given TIC ID and sector.

    Parameters
    ----------
    tic : `str`
        The TIC ID of the target 
    sector_ind : `int`
        The index of the sector to use
    n_inject : `int`
        The number of injections to perform
    n_repeat : `int`
        The number of times to repeat the pipeline
    cache_path : `str`
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
    >>> cvz_pipeline(tic="tic272272592", sector_ind='0', n_inject=2000, n_repeat=10,
                     cache_path="/gscratch/scrubbed/tomwagg/",
                     out_path="/gscratch/dirac/flair/cvz/", cpu_count=10)
    """
    # setup up variables that are checkpointed
    lc = None
    flare_mask, flare_starts, flare_ends = None, None, None
    mu, variance = None, None
    amps, fwhms, insert_points = None, None, None
    recovered, inject_index = None, 0

    # setup the file name and check if it already exists
    file_name = out_path + f"{tic}_{sector_ind}.h5"
    file_exists = isfile(file_name)
    file_keys = None

    # setup the CNN and models
    cnn, models = flair.flares.prep_stella(download_path=None, out_path=out_path)

    # setup the logger with output
    logger = logging.getLogger("flair")
    logger.setLevel(log_level)

    # skip creating the lightcurve and identifying flares if the file already exists
    if file_exists:
        logger.info(f"File already exists for TIC {tic} in sector n={sector_ind}")

        # open the file and read the keys
        with h5.File(file_name, "r") as f:
            file_keys = f.keys()

            # read the lightcurve if it exists
            if "lc" in file_keys:
                lc = lk.LightCurve(time=f["lc/time"][:], flux=f["lc/flux"][:], flux_err=f["lc/flux_err"][:])

                # attach the sector as well and ignore lightkurve warnings about this
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                                            message=".*doesn't allow columns or meta values to be created.*")
                    lc.sector = f["lc"].attrs["sector"]

            # read the flare mask if it exists
            if "flare_mask" in f["lc"]:
                flare_mask = f["lc/flare_mask"][:]
            if "flares" in file_keys:
                flare_starts, flare_ends = f["flares/start_times"][:], f["flares/stop_times"][:]

            # read the GP mean and variance if it exists
            if "gp" in file_keys:
                mu = f["gp/mu"][:]
                variance = f["gp/variance"][:]

            if "injections" in file_keys:
                amps = f["injections/amps"][:]
                fwhms = f["injections/fwhms"][:]
                insert_points = f["injections/insert_points"][:]

            if "recovered" in file_keys:
                recovered = f["recovered"][:]
                inject_index = f["recovered"].attrs["inject_index"]

    # if the lightcurve doesn't exist, download it
    if lc is None:
        logger.info(f"Downloading lightcurve for TIC {tic} in sector n={sector_ind}")

         # set the download cache directory
        lk.conf.cache_dir = cache_path

        # download the lightcurve
        lc = flair.lightcurve.get_lightcurve(target=tic, mission='TESS', author='SPOC', ind=sector_ind)
        
        # CHECKPOINT 1: save the lightcurve
        with h5.File(file_name, "w") as f:
            g = f.create_group("lc")
            g.attrs["sector"] = lc.sector
            g.create_dataset("time", data=lc.time.value)
            g.create_dataset("flux", data=lc.flux.value)
            g.create_dataset("flux_err", data=lc.flux_err.value)

    # if the flare mask doesn't exist, identify flares
    if flare_mask is None:
        logger.info(f"Identifying flares for TIC {tic} in sector n={sector_ind}")
        
        # predict the flares and create a mask
        avg_pred = flair.flares.get_stella_predictions(cnn=cnn, models=models, lc=lc)
        flare_mask, flare_starts, flare_ends = flair.flares.get_flares(flare_prob=avg_pred, min_flare_points=3,
                                                                       merge_absolute=2, merge_relative=0.2)

        # CHECKPOINT 2: save the flare mask
        with h5.File(file_name, "a") as f:
            g = f["lc"]
            g.create_dataset("flare_mask", data=flare_mask)
        with h5.File(file_name, "a") as f:
            g = f.create_group("flares")
            g.create_dataset("start_times", data=flare_starts)
            g.create_dataset("stop_times", data=flare_ends)

        # plot the lightcurve with flares marked
        fig, _ = flair.plot.plot_lc_with_flares(lc, flare_mask, show=False)
        fig.savefig(join(out_path, "plots", f"{tic}_{sector_ind}_flares.png"))

    # if the GP mean and variance don't exist, fit the GP
    if mu is None or variance is None:
        logger.info(f"Fitting GP to lightcurve for TIC {tic} in sector n={sector_ind}")

        # fit the GP to the lightcurve
        opt_gp = flair.gp.fit_GP(lc, flare_mask)
        mu, variance = opt_gp.predict(y=lc.flux.value[~flare_mask], t=lc.time.value, return_var=True)

        # CHECKPOINT 3: save the GP mean and variance
        with h5.File(file_name, "a") as f:
            g = f.create_group("gp")
            g.create_dataset("mu", data=mu)
            g.create_dataset("variance", data=variance)

        _, _, fig, _ = flair.plot.plot_lc_and_gp(lc, mu=mu, variance=variance, flare_mask=flare_mask,
                                                 highlight_flares=False, show=False)
        fig.savefig(join(out_path, "plots", f"{tic}_{sector_ind}_gp.png"))

        # fit Equivalent durations 
        eds= flair.flares.calc_equivalent_durations(lc, flare_starts=flare_starts, flare_ends=flare_ends,
                                                    mu=mu)

        # CHECKPOINT 4: save the EDs with flare starts and stops
        with h5.File(file_name, "a") as f:
            g = f["flares"]
            g.create_dataset("equivalent durations", data=eds)


    if amps is None or fwhms is None or insert_points is None:
        logger.info("Drawing injected flares")

        # flare amplitudes are set by the typical uncertainty in the lightcurve
        norm_median_error = (np.median(lc.flux_err.value) / np.median(lc.flux.value)).tolist()
        amps = np.geomspace(0.5*norm_median_error, 20 * norm_median_error, n_inject)

        # simple choice for FWHM in days
        fwhms = np.linspace(1, 20, n_inject) / 1440

        # draw random injection times
        all_inds = np.arange(len(lc))
        n_end_avoid = 5
        not_flare_inds = all_inds[~flare_mask & (all_inds > n_end_avoid) & (all_inds < len(lc) - n_end_avoid)]
        insert_points = np.random.choice(not_flare_inds, size=(n_repeat, n_inject))

        # CHECKPOINT 5: save the injected flares
        with h5.File(file_name, "a") as f:
            g = f.create_group("injections")
            g.create_dataset("amps", data=amps)
            g.create_dataset("fwhms", data=fwhms)
            g.create_dataset("insert_points", data=insert_points)

    if recovered is None:
        recovered = np.zeros((n_repeat, n_inject), dtype=bool)
    
    i_start = inject_index

    # recovered is a 2D, the len of each row is the number of flares injected
    # the len of each column is the number of times the injection was repeated
    for i in range(i_start, n_repeat):
        logger.info(f"Performing injection for flare {(i)}")
        recovered[i] = flair.inject.injection_test(time=lc.time.value, 
                                                      flux=lc.flux.value,
                                                      flux_err=lc.flux_err.value,
                                                      cnn=cnn,
                                                      models=models,
                                                      flare_mask=flare_mask,
                                                      amp=amps,
                                                      fwhm=fwhms,
                                                      insertion_point=insert_points[i])

        # CHECKPOINT 6: save recovered flares one at a time
        with h5.File(file_name, "a") as f:
            if "recovered" in f.keys():
                d = f["recovered"]
                d[i] = recovered[i]
            else:
                d = f.create_dataset("recovered", data=recovered)
            d.attrs["inject_index"] = i

    logger.info("All done!")

# setup argparse
def main():
    parser = argparse.ArgumentParser(description='Run CVZ pipeline on a specific sector')
    parser.add_argument('-t', '--tic', default="", type=str,
                        help='TIC ID of the star')
    parser.add_argument('-i', '--sector-ind', default="", type=int,
                        help='Alternative to sector, the index of the sector to use (e.g. 2, meaning 2nd sector)')
    parser.add_argument('-ni', '--ninject', default=25, type=int,
                        help='Number of flares to inject')
    parser.add_argument('-nr', '--nrepeat', default=25, type=int,
                        help='Number of times to repeat each injected flare')
    parser.add_argument('-c', '--cache-path', default="/gscratch/scrubbed/tomwagg/", type=str,
                        help='Path to use for downloading lightkurve files')
    parser.add_argument('-o', '--out-path', default="/gscratch/dirac/flair/cvz/", type=str,
                        help='Path to use for output files')
    parser.add_argument('-p', '--cpu-count', default=10, type=int,
                        help='How many CPUs to use for the injection and recovery tests')
    args = parser.parse_args()

    # run the pipeline
    cvz_pipeline(tic=args.tic, sector_ind=args.sector_ind, n_inject=args.ninject,
                 n_repeat=args.nrepeat, cache_path=args.cache_path, out_path=args.out_path,
                 cpu_count=args.cpu_count)


if __name__ == "__main__":
    main()