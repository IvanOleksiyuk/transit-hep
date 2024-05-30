#!/bin/sh
#SBATCH --job-name=twinturbo_reco-cons-aclip
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=1:0:00
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/jobs/job_output/twinturbo_reco-cons-aclip-%A-%x_%a.out
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Job started at: $(date)"

module load GCCcore/12.3.0 Python/3.11.3
cd scratch/sing_images/
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif bash -c "cd /home/users/o/oleksiyu/WORK/hyperproject/ && python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-aclip data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"