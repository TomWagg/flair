import os

# read in the template slurm script
with open("template.slurm") as f:
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
        tic, _, _, _, n_sector = line.split(",")
        tic = f"tic{tic}"

        sectors = [str(i) for i in range(int(n_sector))]# if not os.path.exists(f"../output/{tic}_{i}.h5")]

        sectors = [sectors[0]]

        temp_slurm = template.replace("FLAIR_TIC_ID", tic)
        with open("temp.slurm", "w") as temp:
            temp.write(temp_slurm)

        # submit the job
        os.system(f"sbatch --array={','.join(sectors)} temp.slurm")

        os.remove("temp.slurm")

        # break early for testing
        if n_stars >= N_STAR_LIMIT:
            break