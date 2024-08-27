from pathlib import Path
container: "/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/container/linear.sif"

all_dopes = [0, 667, 1000, 3000]
seeds = [0, 1, 2, 3, 4]
OP_PATH = Path("/srv/beegfs/scratch/users/s/senguptd/purelinear/")
modes = ["standard"]
sb1window=["3100,3300", "3000,3300", "2900,3300", "2800,3300", "2700,3300"]
sb2window=["3700,3900", "3700,4000", "3700,4100", "3700,4200", "3700,4300"]

strip_comma = lambda x: "_".join([str(x) for x in x.split(",")])

def get_input_list(wildcards):
    if wildcards.mode == "standard":
        return "/srv/beegfs/scratch/users/s/senguptd/purelinear/window_{sb1}__{sb2}/dope_{dope}/template.npy"
    elif wildcards.mode in ["idealised", "supervised"]:
        return "/srv/beegfs/scratch/groups/rodem/LHCO/events_anomalydetection_v2.curtains.h5"


rule all:
    input:
        *["/srv/beegfs/scratch/users/s/senguptd/purelinear/" + f"window_{strip_comma(sb1)}__{strip_comma(sb2)}/dope_{dope}/{mode}/seed_{seed}/cwola_outputs.h5" for dope in all_dopes for seed in seeds for mode in modes for sb1,sb2 in zip(sb1window,sb2window)]

rule cwola:
    output:
        "/srv/beegfs/scratch/users/s/senguptd/purelinear/window_{sb1}__{sb2}/dope_{dope}/{mode}/seed_{seed}/cwola_outputs.h5"
    input:
        get_input_list
    log:
        "/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/logs/cwola_{dope}_{seed}_{sb1}_{sb2}_{mode}.log"
    resources:
        runtime=120,
        mem_mb=50000,
    shell:
        "python cwola.py --input_path=/srv/beegfs/scratch/users/s/senguptd/purelinear/ --num_signal={wildcards.dope} --mode={wildcards.mode} --seed={wildcards.seed} --sideband_1={wildcards.sb1} --sideband_2={wildcards.sb2}"

rule template:
    output:
        "/srv/beegfs/scratch/users/s/senguptd/purelinear/window_{sb1}__{sb2}/dope_{dope}/template.npy"
    input:
        "/srv/beegfs/scratch/groups/rodem/LHCO/events_anomalydetection_v2.curtains.h5"
    log:
        "/home/users/s/senguptd/UniGe/Anomaly/lin_an/linearanomaly/logs/template_{dope}_{sb1}_{sb2}.log"
    resources:
        runtime=600,
        mem_mb=50000,
    shell:
        "python export_template.py --output_path=/srv/beegfs/scratch/users/s/senguptd/purelinear/ --num_signal={wildcards.dope} --sideband_1={wildcards.sb1} --sideband_2={wildcards.sb2} --autooversample=0 --pure_sidebands"
