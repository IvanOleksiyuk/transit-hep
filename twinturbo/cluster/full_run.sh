#!/bin/sh
#SBATCH --job-name=CLIP_sm1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=180
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/JetTaggers/logs/slurm-%A-%x_%a.out
#SBATCH --chdir=/home/users/o/oleksiyu/JetTaggers
#SBATCH --mem=32GB
#SBATCH --gres=gpu:ampere:1

module load GCCcore/12.3.0 Python/3.11.3

export XDG_RUNTIME_DIR=""
export NUMEXPR_MAX_THREADS=8
export PYTHONPATH=${PWD}:${PWD}/python_install:${PYTHONPATH}

singularity exec --nv -B /home/users/,/srv,/tmp /home/users/o/oleksiyu/scratch/sing_images/hyperproject.sif bash -c "python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py"