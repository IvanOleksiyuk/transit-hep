#!/bin/sh
#SBATCH --job-name=TT_reco_advII5s_LLV_2048b_lr2-4_dl
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/jobs/job_output/TT_reco_advII5s_LLV_2048b_lr2-4_dl-%A-%x_%a.out
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Job started at: $(date)"

module load GCCcore/12.3.0 Python/3.11.3
cd sing_images/
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif \
 bash -c "cd /home/users/o/oleksiyu/WORK/hyperproject/ &&\
 HYDRA_FULL_ERROR=1 /opt/conda/bin/python /home/users/o/oleksiyu/WORK/hyperproject/libs_snap/anomdiff/scripts/train_disc.py mode=standard"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"