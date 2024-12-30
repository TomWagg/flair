# Read the FLAIR_TIC_ID and n_sector from the CSV file
flair_tic_ids = []
sector_counts = []

with open("../data/TESS_CVZ.csv") as f:
    for line in f:
        if line.startswith("TICID"):
            continue
        tic, _, _, _, n_sector = line.split(",")
        flair_tic_id = f"tic{tic}"
        n_sector = int(n_sector.strip())
        flair_tic_ids.append(flair_tic_id)
        sector_counts.append(n_sector)
        # if len(flair_tic_ids) >= 110:
        #     break

# Write the FLAIR_TIC_ID and SECTOR_ID to the disbatch file
with open("full_disbatch", "w") as f:
    for flair_tic_id, n_sector in zip(flair_tic_ids, sector_counts):
        for sector_id in range(n_sector):
            f.write(f"./exicute.sh {flair_tic_id} {sector_id} &> logs/{flair_tic_id}_{sector_id}.out\n")
    # Write the additional lines at the end of the file
    f.write("#DISBATCH BARRIER\n")
    f.write("./exicute_post.sh &> post.out\n")