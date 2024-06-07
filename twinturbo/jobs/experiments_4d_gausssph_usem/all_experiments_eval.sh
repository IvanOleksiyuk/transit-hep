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
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_only data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-aclip data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr-vic data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-acontr data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-aclip data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_noise_mINx data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_noise_mSEP data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.01a0.02 data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.01a0.05 data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.1both data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.02a0.01 data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.05a0.01 data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco-cons-acontr0.005a0.05 data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_adam data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_DisCo data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
&& python /home/users/o/oleksiyu/WORK/hyperproject/twinturbo/scripts/full_run.py --config-name twinturbo_reco_pearson data=gauss_sph_4_twinturbo_usem general.subfolder=gauss_sph_4_twinturbo_usem/ do_train_template=0 do_export_template=0\
"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"