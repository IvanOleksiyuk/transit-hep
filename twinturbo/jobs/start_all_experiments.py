import os
import subprocess

def submit_slurm_jobs(directory, filename_prefix):
    # Change the working directory to home
    os.chdir(os.path.expanduser("~"))
    
    for filename in os.listdir(directory):
        if filename.startswith(filename_prefix):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                print(f"Submitting job: {filepath}")
                result = subprocess.run(['sbatch', filepath], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Job submitted successfully: {result.stdout}")
                else:
                    print(f"Failed to submit job: {result.stderr}")

if __name__ == "__main__":
    directory = "/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/jobs/experiments_8d_gauss_usem"  # Change this to your specific folder
    filename_prefix = "twinturbo"

    submit_slurm_jobs(directory, filename_prefix)