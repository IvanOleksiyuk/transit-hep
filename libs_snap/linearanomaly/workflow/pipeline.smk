from pathlib import Path
container: "/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/container/linear.sif"
all_dopes = [0, 50, 100, 200, 500, 1000, 3000, 8000]
seeds = [0, 1, 2, 3, 4]
OP_PATH = Path("/home/users/s/senguptd/scratch/linearresults/")
aggregate = "mean"
modes = ["standard", "idealised", "supervised"]
sb1="3100,3300" #,"3000,3300", "2900,3300", "2800,3300", "2700,3300"]
sb2="3700,3900" #,"3700,4000", "3700,4100", "3700,4200", "3700,4300"]
strip_comma = lambda x: "_".join([str(x) for x in x.split(",")])


rule all:
    input:
        *[OP_PATH / f"window_{strip_comma(sb1)}__{strip_comma(sb2)}/dope_{dope}/{mode}/seed_{seed}/fulldata_{aggregate}.h5" for dope in all_dopes for seed in seeds for mode in modes]

rule cwola:
    output:
        output_cwola="/home/users/s/senguptd/scratch/linwindows/window_3100_3300__3700_3900/dope_{{dope}}/{{mode}}/seed_{{seed}}/fulldata_{aggregate}.h5"
    params:
        "cwola.py",
        f"--num_signal={{dope}}",
        f"--seed={{seed}}",
        f"--mode={{mode}}",
        f"--aggregate={aggregate}"
    resources:
        runtime=60,
        mem_mb=20000,
    input:
        f"/home/users/s/senguptd/scratch/linwindows/window_3100_3300__3700_3900/dope_{{dope}}/template.npy"
    log:
        f"/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/logs/cwola_{{dope}}_{{mode}}_{{seed}}.log"
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.5/"

rule template:
    output:
        output_template="/home/users/s/senguptd/scratch/linwindows/window_3100_3300__3700_3900/dope_{dope}/template.npy"
    params:
        "export_template.py",
        f"--num_signal={{dope}}",
    resources:
        runtime=240,
        mem_mb=20000,
    log:
        "/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/logs/template_{dope}.log"
    input:
        "/srv/beegfs/scratch/groups/rodem/LHCO/events_anomalydetection_v2.curtains.h5"
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.5/"
