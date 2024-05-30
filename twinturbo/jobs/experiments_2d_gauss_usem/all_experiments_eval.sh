#!/bin/sh
#SBATCH --job-name=twinturbo_reco_only
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/jobs/job_output/all_eval-%A-%x_%a.out
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

module load GCCcore/12.3.0 Python/3.11.3
cd scratch/sing_images/
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif bash -c \
"cd /home/users/o/oleksiyu/WORK/hyperproject/\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_only do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_only_nom do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-aclip do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr-vic do_train_template=0 do_export_template=0\
"