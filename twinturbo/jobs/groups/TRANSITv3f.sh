#!/bin/sh
#SBATCH --job-name=TRANSITv3f
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/jobs/job_output/TRANSITv3f-%A-%x_%a.out
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Job started at: $(date)"

module load GCCcore/12.3.0 Python/3.11.3
cd sing_images/
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif \
 bash -c "cd /home/users/o/oleksiyu/WORK/hyperproject/ &&\
 python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run_group.py\
 --config-name full_run_group_stability_10.yaml\
 full_run_cfg=TRANSITv3f\
 run_dir=workspaces/ML4Jets/TRANSITv1"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"