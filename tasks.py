import glob
import logging
import os
import re
from pathlib import Path
from typing import Optional

import yaml
from invoke import Context, task

# setup logger
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


@task
def experiment_run(
    ctx: Context,
    name: str,
    group: Optional[str] = None,
    workflow: str = "main",
    profile: str = "slurm",
    stage: Optional[str] = None,
    use_orig_configs: bool = True,
    snakemake_args: str = "",
):
    """Run an experiment through a snakemake workflow.

    All output will be gathered in experiments/<group>/<name>/. Persistent defaults for
    group and stage can be found in config/common/default.yaml and
    config/common/private.yaml.

    Args
    ----
        ctx (Context): invoke context
        name (str): name of the experiment, used for identification and to determine the
                    output directory.
        group (Optional[str], optional): group of the experiment, used for
                                         identification and to determine the output
                                         directory (default: None).
        workflow (str, optional): name of the workflow, available workflows are located
                                  in the workflow directory under `<workflow>.smk`
                                  (default: "main").
        profile (str, optional): workflow profile, sets hardware specific snakemake
                                 flags , available profiles are located in the
                                 workflow/profiles directory (default: "test").
        stage (Optional[str], optional): experiment stage, should be either "debug" or
                                         "experiment". Used for wandb tags
                                         (default:None).
        snakemake_args (str, optional): anything additional one might want to pass to
                                        the snakemake command (default: "").
    """
    workflows = [Path(p).stem for p in glob.glob("workflow/*.smk")]
    if workflow not in workflows:
        log.error(
            f'Workflow "{workflow}" not found.'
            f' Available workflows: {", ".join(workflows)}'
        )
        return

    profiles = os.listdir("workflow/profiles")
    if profile not in profiles:
        log.error(
            f'Profile "{profile}" not found.'
            f' Available profiles: {", ".join(profiles)}'
        )
        return

    snakemake_cfg = [f"experiment_name={name}"]
    if stage:
        snakemake_cfg.append(f"stage={stage}")
    if group:
        snakemake_cfg.append(f"experiment_group={group}")
    if use_orig_configs:
        snakemake_cfg.append(f"use_orig_configs={use_orig_configs}")
    snakemake_cfg = " ".join(snakemake_cfg)

    snakemake_cmd = [
        "snakemake",
        f"--snakefile workflow/{workflow}.smk",
        f"--workflow-profile workflow/profiles/{profile}",
        f"--config {snakemake_cfg}",
    ]
    snakemake_cmd = " ".join(snakemake_cmd)

    ctx.run(f"{snakemake_cmd} {snakemake_args}", pty=True)


@task
def pre_commit(ctx: Context):
    ctx.run("pre-commit run --all-files", pty=True)


@task
def run_unit_tests(ctx: Context):
    # for some reason just running pytest without `python -m` doesn't work
    ctx.run("python -m pytest tests", pty=True)


@task
def container_pull(ctx: Context, login: bool = False):
    apptainer_pull_args = []
    if login:
        apptainer_pull_args.append("--docker-login")
    apptainer_pull_args = " ".join(apptainer_pull_args)

    origin_url = ctx.run("git remote get-url origin", hide=True).stdout.strip()
    container_url = get_container_url_from_origin_url(origin_url)

    with open("config/common/private.yaml") as f:
        container_path = yaml.safe_load(f)["container_path"]

    container_path_parent = Path(container_path).parent
    if not os.path.exists(container_path_parent):
        os.makedirs(container_path_parent, exist_ok=True)

    if login:
        log.info("Please enter the credentials of your CERN computing account.")

    log.info("runnung: "+f"apptainer pull {apptainer_pull_args} {container_path} {container_url}")
    return_code = ctx.run(
        f"apptainer pull {apptainer_pull_args} {container_path} {container_url}",
        pty=True,
        warn=True,
    ).return_code

    if return_code != 0:
        log.error(
            "Failed to pull the container, you might want to try again with `--login`."
        )


def get_container_url_from_origin_url(origin_url: str):
    (project_path,) = re.match(
        r"[a-z]+://[^/]*gitlab.cern.ch[^/]*/(.+)", origin_url
    ).groups(1)
    project_path = project_path.removesuffix("/").removesuffix(".git")

    return f"docker://gitlab-registry.cern.ch/{project_path}/docker-image:latest"
