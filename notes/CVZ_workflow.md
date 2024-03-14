# Workflow for CVZ processing
(That's pronounced see-vee-zed folks ;P)

## Inputs
$N_* \sim \mathcal{O}(20,000)$ TIC IDs - every continuous viewing zone star (defined as at least 10 sectors per year) that matches our sample
- Each star will have $N_{\rm LC} \sim \mathcal{O}(10)$ lightcurves with $N_{\rm t} \sim \mathcal{O}(22,000)$ timesteps (estimated with 30 days of 2 minute cadence).
- Total input file size is therefore $\sim \mathcal{O}(80 {\rm GB})$ (assuming 20 columns per lightcurve file)

## Pipeline
Allocate a Hyak job for each sector and run them on the checkpoint queue on Klone. Checkpoint jobs can be cancelled at any time and *will* be re-queued every 4 hours. Therefore, this pipeline includes multiple instances at which the code is checkpoint-ed so we can skip to that point if the job is re-queued.

1. Download the lightcurve for the sector using LightKurve
    - Save to `/gscratch/scrubbed` - it gets wiped every 21 days but we can save the parts of the lightcurve we need elsewhere, don't need every column
2. Run `stella` on the lightcurve with all of its models to get flare probabilities
3. Turn flare probabilities into a flare mask based on our criteria (consecutive points, merging adjacent flares etc.)
    - **(CHECKPOINT 1)** Store lightcurve and flare mask to `{TIC_ID}_{sector}.h5` under key `lc`
    - Save in `gscratch/dirac`, seems like there's lots of space there (they have 1600 Gb free)
4. Fit Gaussian process to the lightcurve with flares masked out
    - **(CHECKPOINT 2)** Store $\mu, \sigma$ of the GP in the same `lc` table
5. Perform injection and recovery tests 
    - Set attribute `n_inject_remaining = N_inject` in under group `recoveries`
    - Repeat $N_{\rm inject}$ times
        - Select (or sample) a flare with some amplitude, $A$ and duration, $\tau$
        - Repeat $N_{\rm repeat}$ times, using multiprocessing to parallelise this
            - Inject the flare at a random non-flaring point in the lightcurve, run `stella` on a window around that point, save whether flare is identified
        - **(CHECKPOINT 3)** Save boolean array of recovery as new row in under key `recoveries`. Update attribute `n_inject_remaining -= 1`. Save flare duration, amplitude and injection times as a new row in under key `injections`

### Runtime
The runtime of each sector will depend on exactly how many timesteps there are in the lightcurve. Assuming the download speed is high, the time to checkpoint 1 should be very short. Time to checkpoint 2 will likely take around 1.5 minutes. Therefore, we can anticipate that the majority of the runtime comes from the injection and recovery.

Runtime = $\mathcal{O}(N_{\rm inject} \cdot (N_{\rm repeat} \cdot N_{\rm core}^{-1} + t_{\rm Pool}) \cdot 3 s)$

Assuming we inject 2000 flares, 10 times each and use 10 cores, each job should run in $\mathcal{O}(2 \, {\rm hours})$ - well inside the re-queue time.

## Outputs
Each sector will output a single HDF5 file labelled `{TIC_ID}_{sector}.h5` to `/gscratch/dirac` containing 3 groups

**`lc`**
- The lightcurve (time, flux, flux_err)
- Boolean mask of whether each timestep is part of a flare
- Mean and variance of the GP at each timestep
- Expected size: $\mathcal{O}(N_{\rm timestep} \cdot [{\rm 5 \cdot sizeof(float) + 2 \cdot sizeof(bool)}]) = \mathcal{O}(\rm 1 Mb)$

**`recoveries`**
- 2D ($N_{\rm inject}$ x $N_{\rm repeat}$) boolean array of recoveries, indexed on `flare_id`
- Expected size: $\mathcal{O}(N_{\rm inject} \cdot N_{\rm repeat} \cdot {\rm sizeof(bool)}) = \mathcal{O}(\rm 20 kb)$


**`injections`**
- Flare durations and amplitudes and the times at which they were injected, indexed on `flare_id`
- Expected size: $\mathcal{O}(N_{\rm inject} \cdot [{\rm 2 \cdot sizeof(float)} + N_{\rm repeat} \cdot {\rm sizeof(float)}]) = \mathcal{O}(\rm 1 Mb)$

So each file is only on the order of a couple of Mb so we shouldn't need to worry about storage issues.