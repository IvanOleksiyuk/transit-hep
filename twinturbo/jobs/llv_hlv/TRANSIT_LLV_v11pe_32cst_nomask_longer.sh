#!/bin/sh
#SBATCH --job-name=TRANSIT_LLV_v11pe_32cst_nomask
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/jobs/job_output/TRANSIT_LLV_v11pe_32cst_nomask-%A-%x_%a.out
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Job started at: $(date)"

module load GCCcore/12.3.0 Python/3.11.3
cd sing_images/
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif \
 bash -c "cd /home/users/o/oleksiyu/WORK/hyperproject/ &&\
 python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py\
 --config-name TRANSIT_LLV_v11pe_32cst_nomask general.subfolder=TRANSIT_LLV/\
 verbose_validation=1"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"