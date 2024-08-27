from pathlib import Path
container: "/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/container/linear.sif"
all_dopes = [0, 500, 3000, 8000]
seeds = [0, 1, 2, 3, 4]
sideband1 = ['2700,3300', '2900,3300', '3000,3300', '3100,3300', '3200,3300']
sideband2 = ['3700,4300', '3700,4100', '3700,4000', '3700,3900', '3700,3800']
OP_PATH = Path("/home/users/s/senguptd/scratch/linear_bump/")
aggregate = "mean"

rule all:
    input:
        *[OP_PATH / f"band_{sb1}_{sb2}/dope_{dope}/seed_{seed}/fulldata_{aggregate}.h5" for sb1,sb2 in zip(sideband1, sideband2) for dope in all_dopes for seed in seeds]

rule cwola:
    output:
        f"{OP_PATH}/band_{{sb1}}_{{sb2}}/dope_{{dope}}/seed_{{seed}}/fulldata_{aggregate}.h5"
    params:
        "cwola.py",
        f"--num_signal={{dope}}",
        f"--seed={{seed}}",
        f"--sideband_1={{sb1}}",
        f"--sideband_2={{sb2}}",
        f"--aggregate={aggregate}"
    resources:
        runtime=60,
        mem_mb=20000,
    input:
       f"{OP_PATH}/band_{{sb1}}_{{sb2}}/dope_{{dope}}/template.npy"
    log:
        f"/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/logs/cwola_band_{{sb1}}_{{sb2}}_{{dope}}_{{seed}}.log"
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.5/"

rule template:
    output:
        f"{OP_PATH}/band_{{sb1}}_{{sb2}}/dope_{{dope}}/template.npy"
    params:
        "export_template.py",
        f"--num_signal={{dope}}",
        f"--sideband_1={{sb1}}",
        f"--sideband_2={{sb2}}"
    resources:
        runtime=60,
        mem_mb=20000,
    log:
        f"/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/logs/template_band_{{sb1}}_{{sb2}}_{{dope}}.log"
    input:
        "/srv/beegfs/scratch/groups/rodem/LHCO/events_anomalydetection_v2.curtains.h5"
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.5/"
