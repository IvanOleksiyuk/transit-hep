<div align="center">

# TRANSIT-HEP

by Ivan Oleksiyuk (ivan.oleksiyuk@gmail.com)

[![python](https://img.shields.io/badge/-Python_3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/-PyTorch_2.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.1-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra_1.3-89b8cd&logoColor=white)](https://hydra.cc/)
[![snakemake](https://img.shields.io/badge/-Snakemake_7.32.4-039475)](https://snakemake.readthedocs.io/)
</div>

## How to reproduce the plots in publication

### 0 Setup

1. Clone the repository
2. Create an singularity/docker image using Dockerfile OR create a conda environment with requirements.txt

### 1 Get and preprocess data

Download the LHCO R&D dataset "events_anomalydetection_v2.features.h5" from https://zenodo.org/record/4536377
Use /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/preprocess_data.py to preprocess the data

### 2 Generate configs for all experiments

Go in the singularity image and run:
```
HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py\
 --config-name full_run_group_dopings_6seeds.yaml\
 full_run_cfg=TRANSITv0v2_LHCO\
 run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_dopings\
 do_stability_analysis: False\
 one_run_sh=/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/PAPER/TRANSITv0v2_LHCO_one_run.sh

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py\
 --config-name full_run_group_stability_30.yaml\
 full_run_cfg=TRANSITv0v2_LHCO\
 run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_group\
 do_stability_analysis: False\
 one_run_sh=/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/PAPER/TRANSITv0v2_LHCO_one_run.sh
```

### 3 Run all the experiments in parallel using slurm

In the cluster terminal run
```
python submit.py /home/users/o/oleksiyu/WORK/hyperproject/workspaces/PAPER/TRANSITv0v2_LHCO_dopings
python submit.py /home/users/o/oleksiyu/WORK/hyperproject/workspaces/PAPER/TRANSITv0v2_LHCO_group
```
Wait unttill all the jobs are finished

### 4 Gather the experiment results

Go in the singularity image and run:
```
HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py\
 --config-name full_run_group_dopings_6seeds.yaml\
 full_run_cfg=TRANSITv0v2_LHCO\
 run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_dopings\
 one_run_sh=/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/PAPER/TRANSITv0v2_LHCO_one_run.sh

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py\
 --config-name full_run_group_stability_30.yaml\
 full_run_cfg=TRANSITv0v2_LHCO\
 run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_group\
 one_run_sh=/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/PAPER/TRANSITv0v2_LHCO_one_run.sh
```

### 5 Produce plots 

Run transit/notebooks/final_plotsPAPER.ipynb with Jupyter

## License

This project is licensed under the MIT License. See the LICENSE file for details.
