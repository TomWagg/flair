# Flare rates

Pipeline to Calculate Flare Frequency Distributions for stars in the TESS Continuous Viewing Zone. 

Code which does the calculations in in the flair repo, and put together in the pipeline repo. The `cvz.py`
function is the script which runs everything. The script runs `stella`, masks out found flares, fits a GP and calculates the equivalent durations for found flares. We then run injection and recovery of synthetic flares to determine and correct the completeness. Code is set up to run on a single TESS sector, of a single star, at a time. The output is a hdf5 file with the light curve, the found flares, the injection and recovery results and the equivalent durations. 

The slurm directory contains the scripts necessary to run the pipeline on clone. Currently set up to run in Tobin's gscratch folder. 