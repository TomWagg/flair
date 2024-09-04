import os
import re
import subprocess

log_dir = '/mmfs1/gscratch/dirac/flair/slurm/logs/'  # Directory containing the log files

# List to store .err files
err_files = []

# Loop through every file in the directory
for log_file in os.listdir(log_dir):
    if log_file.endswith('.err'):
        err_files.append(log_file)

# Regular expression to extract tic and sector
tic_sector_pattern = re.compile(r'Running on tic(\d+), sector (\d+)')

tics_and_sectors = []

# Check each error log file
for err_file in err_files:
        with open(os.path.join(log_dir, err_file), 'r') as f:
            content = f.read()
            if 'error' in content.lower():  # Check if 'error' is in the content
                with open(os.path.join(log_dir, err_file[:-4]+'.out'), 'r') as h:
                    first_line = h.readline()
                    match = tic_sector_pattern.match(first_line)
                    if match:
                        tic_id = match.group(1)
                        sector = match.group(2)
                        tics_and_sectors.append((tic_id, sector))

# Define the path to the slurm script
slurm_script_path = '/mmfs1/gscratch/dirac/flair/slurm/cvz.slurm'

# Read the original slurm script
with open(slurm_script_path, 'r') as file:
    slurm_script_content = file.read()

# Function to modify the slurm script for a specific job
def modify_slurm_script(tic, sector, content):
    # Replace FLAIR_TIC_ID with the tic
    content = content.replace('$FLAIR_TIC_ID', str(tic))
    # Replace SLURM_ARRAY_TASK_ID with the sector
    content = content.replace('$SLURM_ARRAY_TASK_ID', str(sector))
    content = re.sub(r'echo "Running on \${FLAIR_TIC_ID}, sector \${SLURM_ARRAY_TASK_ID}"', 
                     f'echo "Running on {tic}, sector {sector}"', content)
    # Increase time and memory (example: increase time by 2 hours and memory by 2GB)
    content = re.sub(r'#SBATCH --time=\d+:\d+:\d+', '#SBATCH --time=02:00:00', content)
    content = re.sub(r'#SBATCH --mem=\d+G', '#SBATCH --mem=125', content)
    # Remove the "&& python check_failed.py" part
    content = re.sub(r'&& python check_failed.py', '', content)  
    return content

# Process each failed job
if len(tics_and_sectors) > 0 :
    for tic, sector in tics_and_sectors:
        # Modify the slurm script for the current job
        modified_content = modify_slurm_script(f'tic{tic}', sector, slurm_script_content)
        # Write the modified content to a new temporary file
        temp_slurm_script_path = f'/mmfs1/gscratch/dirac/flair/slurm/temp_cvz_{tic}_{sector}.slurm'
        with open(temp_slurm_script_path, 'w') as temp_file:
            temp_file.write(modified_content)

        if os.path.exists(temp_slurm_script_path) and os.path.getsize(temp_slurm_script_path) > 0:
            # Submit the temporary slurm file
            os.system(f'sbatch {temp_slurm_script_path}')
            # Delete the temporary slurm file
            os.remove(temp_slurm_script_path)
        else:
            print(f"File {temp_slurm_script_path} does not exist or is empty")


else:
    print('No found Errors')