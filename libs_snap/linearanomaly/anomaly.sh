#!/bin/sh

#SBATCH --cpus-per-task=6
#SBATCH --mem=16GB
#SBATCH --time=0-01:00:00
#SBATCH --job-name=anomaly
#SBATCH --output=/home/users/s/senguptd/UniGe/Anomaly/linearanomaly/logs/%A_%a.out
#SBATCH --chdir=/home/users/s/senguptd/UniGe/Anomaly/linearanomaly
#SBATCH --partition=shared-cpu,private-dpnc-cpu
#SBATCH -a 0-44

num_signal=( 0 50 100 250 500 1000 1500 2000 2500 3000 )
seed=( 0 1 2 3 4 )

export XDG_RUNTIME_DIR=""
srun apptainer exec --nv -B /srv,/home \
   /srv/fast/share/rodem/images/jetssl_latest.sif \
   python run.py \
       --num_signal=${num_signal[`expr ${SLURM_ARRAY_TASK_ID} / 1 % 7`]} \
       --seed=${seed[`expr ${SLURM_ARRAY_TASK_ID} / 7 % 5`]} \
       --preprocessor=minmax \
