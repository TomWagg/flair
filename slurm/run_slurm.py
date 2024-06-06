import os
from copy import copy

# read in the template slurm script
with open("cvz.slurm") as f:
    template = f.read()

n_stars = 0
N_STAR_LIMIT = 1

with open("../data/TESS_CVZ.csv") as f:
    # go through the CVZ file
    for line in f:
        if line.startswith("TICID"):
            continue

        n_stars += 1
            
        # read in the star
        tic, _, _, _, n_sector = f.readline().split(",")
        tic = f"tic{tic}"

        sectors = [str(i) for i in range(int(n_sector)) if not os.path.exists(f"../output/{tic}_{i}.h5")]

        os.setenv("FLAIR_TIC_ID", tic)
                
        # submit the job
        os.system(f"sbatch --array={",".join(sectors)} cvz.slurm")

        # break early for testing
        if n_stars >= N_STAR_LIMIT:
            break
