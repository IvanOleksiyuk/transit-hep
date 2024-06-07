#!/bin/sh
#SBATCH --job-name=all_eval
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/jobs/job_output/all_eval-%A-%x_%a.out
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Job started at: $(date)"

module load GCCcore/12.3.0 Python/3.11.3
cd scratch/sing_images/
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif bash -c \
"cd /home/users/o/oleksiyu/WORK/hyperproject/\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_only do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-aclip do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr-vic do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-acontr do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-aclip do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_noise_mINx do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_noise_mSEP do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.01a0.02 do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.01a0.05 do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.1both do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.02a0.01 do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.05a0.01 do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.005a0.05 do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_adam do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_DisCo do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_pearson do_train_template=0 do_export_template=0\
"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"