from pathlib import Path
container: "/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/container/linear.sif"


seeds = [0, 1, 2, 3, 4]

curtains_path = Path("/srv/beegfs/scratch/users/s/senguptd/curtains/images/doped_lhco/")
jobs = list(curtains_path.glob("*"))
jobs.sort(key = lambda x: int(x.name.split("_")[-1]))

def get_input_list(wildcards):
    job = wildcards.job
    return Path(job)/"evaluation/samples_sb1_2_to_sr.npz"

aggregate = "mean"
modes = ["standard"] #, "idealised", "supervised"]


rule all:
    input:
        *[job/f"bdt/{mode}/seed_{seed}/fulldata_{aggregate}.h5" for job in jobs for seed in seeds for mode in modes]

rule cwola:
    output:
        "{job}/bdt/{mode}/seed_{seed}/fulldata_mean.h5"
    input:
        get_input_list
    log:
        "/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/logs/curtains_cwola_{job}_{mode}_{seed}.log"
    resources:
        runtime=120,
        mem_mb=50000,
    shell:
        "python curtains_cwola.py --curtains_path={wildcards.job} --mode={wildcards.mode} --seed={wildcards.seed} --aggregate=mean"
