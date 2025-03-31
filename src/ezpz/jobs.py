"""
jobs.py
"""

from __future__ import absolute_import, annotations, division, print_function
import logging
import os
import json
import yaml
from pathlib import Path
from typing import Optional, Any, Union
from ezpz.dist import get_rank
from ezpz.configs import (
    get_scheduler,
)

RANK = get_rank()
SCHEDULER = get_scheduler()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FROM_ALL_RANKS = os.environ.get("LOG_FROM_ALL_RANKS", False)

log = logging.getLogger(__name__)
if LOG_FROM_ALL_RANKS:
    log.setLevel(LOG_LEVEL)
else:
    log.setLevel(LOG_LEVEL) if RANK == 0 else log.setLevel("CRITICAL")


def check_scheduler(scheduler: Optional[str] = None) -> bool:
    from ezpz.configs import SCHEDULERS

    scheduler = SCHEDULER if scheduler is None else scheduler
    if scheduler is not None and len(scheduler) > 0:
        assert scheduler.upper() in SCHEDULERS.values()
    if scheduler.lower() != "pbs":
        raise TypeError(f"{scheduler} not yet implemented!")
    return True


# def get_jobdir_from_env(scheduler: Optional[str] = None) -> Path:
def get_jobdir_from_env() -> Path:
    from ezpz.dist import get_pbs_env

    pbs_env = get_pbs_env()
    hostfile = os.environ.get(
        "HOSTFILE",
        os.environ.get("PBS_NODEFILE", os.environ.get("HOSTFILE", None)),
    )
    # jobid = pbs_env["PBS_JOBID"].split('.')[0]
    jobdir = Path.home() / f"{SCHEDULER}-jobs" / f"{jobid}"
    jobdir.mkdir(exist_ok=True, parents=True)
    return jobdir


def get_jobid() -> str:
    jobenv = get_jobenv()
    return jobenv["PBS_JOBID"].split(".")[0]


def get_jobfile_ext(ext: str) -> Path:
    # jobid = get_jobid()
    jobdir = get_jobdir_from_env()
    # return jobdir.joinpath(f'{SCHEDULER}-{jobid}.{ext}')
    return jobdir.joinpath(f"jobenv.{ext}")


def get_jobfile_sh() -> Path:
    jobfile_sh = get_jobfile_ext("sh")
    jobfile_sh.parent.mkdir(exist_ok=True, parents=True)
    return jobfile_sh


def get_jobfile_yaml() -> Path:
    jobfile_ext = get_jobfile_ext("yaml")
    jobfile_ext.parent.mkdir(exist_ok=True, parents=True)
    return jobfile_ext


def get_jobfile_json() -> Path:
    jobfile_ext = get_jobfile_ext("json")
    jobfile_ext.parent.mkdir(exist_ok=True, parents=True)
    return jobfile_ext


def get_jobenv(
    framework: Optional[str] = None,
    hostfile: Optional[Union[str, Path]] = None,
    # max_hosts_to_print: Optional[int] = None,
    verbose: Optional[bool] = None,
    verbose_dist_info: Optional[bool] = None,
    verbose_pbs_env: Optional[bool] = None,
) -> dict:
    from ezpz.dist import (
        # get_pbs_launch_info,
        get_dist_info,
        get_pbs_env,
        # get_pbs_launch_cmd,
    )

    jobenv: dict[str, str | int | list[Any]] = get_dist_info(
        hostfile=hostfile,
        framework=framework,
        verbose=verbose_dist_info,
    )
    if SCHEDULER.lower() == "pbs":
        # from ezpz.dist import get_pbs_launch_cmd
        # if (dlaunch := os.environ.get("DIST_LAUNCH", None)) is not None:
        #     dinfo |= {"DIST_LAUNCH": dlaunch}
        jobenv |= get_pbs_env(hostfile=hostfile, verbose=verbose_pbs_env)
        # jobenv |= get_pbs_launch_info(hostfile=hostfile)
        # jobenv |= {'LAUNCH_CMD': get_pbs_launch_cmd(hostfile=hostfile)}
        if verbose:
            log.info(
                "\n".join(
                    ["\n", "[JOB ENV]:"]
                    + [f"  • {k}={v}" for k, v in jobenv.items()]
                    + ["\n"]
                )
            )
        # jobenv |= get_dist_info(
        #     framework=framework,
        #     verbose=verbose,
        #     max_hosts_to_print=max_hosts_to_print,
        #     hostfile=hostfile,
        # )
        return jobenv
    # TODO: Add Slurm support to Python API
    raise ValueError(f"{SCHEDULER} not yet implemented!")


def get_jobslog_file() -> Path:
    jobslog_file = Path.home().joinpath(f"{SCHEDULER}-jobs.log")
    jobslog_file.parent.mkdir(exist_ok=True, parents=True)
    return jobslog_file


def add_to_jobslog(hostfile: Optional[Union[str, Path]] = None):
    jobenv = get_jobenv(hostfile=hostfile)
    assert len(jobenv.keys()) > 0
    jobdir = get_jobdir_from_env()
    assert jobenv is not None
    assert jobdir is not None
    jobslog_file = get_jobslog_file()
    # jobfile_sh = get_jobfile_sh()
    # jobfile_yaml = get_jobfile_yaml()
    Path(jobslog_file).parent.mkdir(exist_ok=True, parents=True)
    last_jobdir = get_jobdir_from_jobslog(-1)
    if jobdir.as_posix() != last_jobdir:
        with jobslog_file.open("a") as f:
            _ = f.write(f"{jobdir}\n")
    else:
        log.warning(
            " ".join(
                [
                    f"{jobdir.as_posix()} ",
                    f"already in {jobslog_file.as_posix()}, ",
                    f"not appending !!",
                ]
            )
        )


def save_to_dotenv_file(
    jobenv: Optional[dict[str, str]] = None,  # type:ignore[reportDeprecated]
    hostfile: Optional[Union[str, Path]] = None,
    verbose: Optional[bool] = None,
) -> Path:
    jobenv = get_jobenv(hostfile=hostfile) if jobenv is None else jobenv
    denvf1 = Path(get_jobdir_from_env()).joinpath(".jobenv")
    denvf2 = Path(os.getcwd()).joinpath(".jobenv")
    # launch_cmd = jobenv.get('LAUNCH_CMD', get_pbs_launch_cmd)
    from ezpz.dist import get_pbs_launch_cmd

    launch_cmd = get_pbs_launch_cmd(hostfile=hostfile)
    assert launch_cmd is not None
    for denvf in [denvf1, denvf2]:
        log.info(
            " ".join(
                ["Saving job env to", f"{denvf.parent.as_posix()}/.jobenv"]
            )
        )
        # launch_cmd = jobenv.get(
        #         'LAUNCH_CMD',
        #         get_jobenv().get('LAUNCH_CMD', '')
        # )
        # if launch_cmd is not None:
        # _ = f.write(f'echo "creating alias launch={launch_cmd}"\n')
        with denvf.open("w") as f:
            _ = f.write("#!/bin/bash --login\n")
            for key, val in jobenv.items():
                _ = f.write(f'{key.upper()}="{val}"\n')
            _ = f.write(f'alias launch="{launch_cmd}"\n')
            _ = f.write(f'echo "$(which launch)"\n')
    # log.warning(' '.join([
    #     f'To use `launch` alias, be sure to: ',
    #     f'`source {denvf2.as_posix()}'
    # ]))
    if verbose:
        log.critical(
            "\n".join(
                [
                    "",
                    "Run:",
                    "",
                    f"    source {denvf2.relative_to(os.getcwd()).as_posix()}",
                    "",
                    "to set environment variables.",
                    "",
                    # "Then, running :"
                    # "    run_cmd=${LAUNCH_CMD} <cmd>\"'",
                    # "    3. 'eval \"${run_cmd}\"'",
                    # # "2. 'launch <cmd>'",
                    # # "  or, 'eval ${LAUNCH_CMD}'",
                    # # "3. , 'echo ${LAUNCH_CMD}'",
                    # "",
                    # f"{name}='{lcmd}'",
                    'Then, running `echo "${LAUNCH_CMD}"` should print:',
                    "",
                    f"    {launch_cmd}",
                    "",
                    # "  ()"
                ]
            )
        )
    return denvf2


def write_pbs_launch_shell_script():
    contents = f"""#!/bin/bash --login
    # This script is used to set up the environment for running jobs on a PBS system.
    source <(curl 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh')
    ezpz_setup_env
    launch python3 -m ezpz.test_dist
    """
    return contents


def write_launch_shell_script():
    contents = """
    #!/bin/bash --login\n
    \n
    alias launch="${LAUNCH}"\n
    echo $(which launch)\n
    \n
    function ezLaunch() {\n
        launch "$@"\n
    }\n
    """
    local_bin = Path().home().joinpath(".local", "bin")
    local_bin.mkdir(exist_ok=True, parents=True)
    launch_file = local_bin.joinpath("launch.sh")
    # launch_file.chmod(launch_file.stat().st_mode | stat.S_IEXEC)
    log.info(f"Saving launch command to {launch_file} and adding to PATH")
    with launch_file.open("w") as f:
        _ = f.write(contents)
    os.chmod(path=launch_file, mode=755)
    path = os.environ.get("PATH")
    path = f"{path}:$HOME/.local/bin"
    os.environ["PATH"] = f"{path}"


def savejobenv_sh(
    jobenv: Optional[dict[str, str]] = None,  # type:ignore[reportDeprecated]
) -> dict[str, str]:
    jobenv = get_jobenv() if jobenv is None else jobenv
    jobfile_sh = get_jobfile_sh()
    jobenv |= {"jobfile_sh": jobfile_sh.as_posix()}
    launch_cmd = jobenv.get("LAUNCH_CMD")
    log.info(f"Saving job env to {jobfile_sh}")
    with jobfile_sh.open("w") as f:
        _ = f.write("#!/bin/bash --login\n")
        for key, val in jobenv.items():
            _ = f.write(f'export {key.upper()}="{val}"\n')
        if launch_cmd is not None:
            _ = f.write(f'alias launch="{launch_cmd}"')
    return jobenv


def savejobenv_json(
    jobenv: Optional[dict[str, str]] = None,  # type:ignore[reportDeprecated]
) -> dict[str, str]:
    jobenv = get_jobenv() if jobenv is None else jobenv
    assert len(jobenv.keys()) > 0
    jobfile_json = get_jobfile_json()
    jobenv |= {"jobfile_json": jobfile_json.as_posix()}
    log.info(f"Saving job env to {jobfile_json}")
    with jobfile_json.open("w") as f:
        json.dump(json.dumps(jobenv, indent=4), f)
    return jobenv


def savejobenv_yaml(
    jobenv: Optional[dict[str, str]] = None,  # type:ignore[reportDeprecated]
) -> dict[str, str]:
    jobenv = get_jobenv() if jobenv is None else jobenv
    assert len(jobenv.keys()) > 0
    jobfile_yaml = get_jobfile_yaml()
    jobenv |= {"jobfile_yaml": jobfile_yaml.as_posix()}
    log.info(f"Saving job env to {jobfile_yaml}")
    with jobfile_yaml.open("w") as f:
        yaml.dump(jobenv, f)
    return jobenv


# def get_launch_cmd(
#         verbose: bool = True,
#         ngpus: Optional[int] = None,
#         nhosts: Optional[int] = None,
#         ngpu_per_host: Optional[int] = None,
#         hostfile: Optional[Union[str, os.PathLike, Path]] = None,
# ):
#     # from ezpz.dist import get_pbs_launch_cmd
#     name = "LAUNCH_CMD"
#     lcmd = (jobenv := get_jobenv()).get(
#         name,
#         jobenv.get(name, jobenv.get(name.lower()))
#         # , os.environ.get("DIST_LAUNCH", None))
#     )
#     # if lcmd is None:
#     #     name = "DIST_LAUNCH"
#     #     lcmd = os.environ.get("DIST_LAUNCH", None)
#     if lcmd is not None and verbose:
#         log.critical('\n'.join([
#             "",
#             "Run:",
#             "",
#             "    source ./.jobenv",
#             "",
#             "to set environment variables.",
#             "",
#             # "Then, running :"
#             # "    run_cmd=${LAUNCH_CMD} <cmd>\"'",
#             # "    3. 'eval \"${run_cmd}\"'",
#             # # "2. 'launch <cmd>'",
#             # # "  or, 'eval ${LAUNCH_CMD}'",
#             # # "3. , 'echo ${LAUNCH_CMD}'",
#             # "",
#             # f"{name}='{lcmd}'",
#             "Then, running `echo \"${LAUNCH_CMD}\"` should print:",
#             "",
#             f"    {lcmd}",
#             "",
#             # "  ()"
#         ]))
#         # md = Markdown(md_str)
#         # console.print(md)
#     return lcmd


def savejobenv(
    verbose: Optional[bool] = None,
    framework: Optional[str] = None,
    hostfile: Optional[Union[str, Path]] = None,
    # max_hosts_to_print: Optional[int] = None,
    print_jobenv: Optional[bool] = None,
    verbose_dotenv: Optional[bool] = None,
    verbose_get_jobenv: Optional[bool] = None,
    verbose_dist_info: Optional[bool] = None,
    verbose_pbs_env: Optional[bool] = None,
):
    jobenv: dict[str, Any] = get_jobenv(
        verbose=verbose_get_jobenv,
        hostfile=hostfile,
        framework=framework,
        # max_hosts_to_print=max_hosts_to_print,
        verbose_pbs_env=verbose_pbs_env,
        verbose_dist_info=verbose_dist_info,
    )
    assert len(jobenv.keys()) > 0
    # assert jobenv is not None
    dotenv_file = save_to_dotenv_file(
        jobenv=jobenv, hostfile=hostfile, verbose=verbose_dotenv
    )
    # jobid = get_jobid()
    pbs_jobid = os.environ.get("PBS_JOBID")
    pbs_nodefile = Path(os.environ.get("PBS_NODEFILE", ""))
    if hostfile is None and pbs_jobid is not None and pbs_nodefile.is_file:
        jobdir = get_jobdir_from_env()
        assert jobdir is not None
        log.info(
            " ".join(
                [
                    f"Caught {pbs_jobid=}, {pbs_nodefile=} from env.",
                    "Saving jobenv!",
                ]
            )
        )
        # -------------------------------------------------------------------
        # Append {jobdir} as a new line at the end of ~/{scheduler}-jobs.log
        # where:
        #   jobdir = Path.home() / f'{scheduler}-jobs' / f'{jobid}'
        add_to_jobslog()
        # -------------------------------------------------------------------
        # Save {scheduler}-related environment variables to
        # `{.sh,.yaml,.json}` files INSIDE {jobdir}
        # for easy loading in other processes
        log.info(
            " ".join(
                [
                    f"Writing {SCHEDULER} env vars to ",
                    f"{jobdir} / jobenv" + "{.sh, .yaml, .json}",
                ]
            )
        )
        jobenv = savejobenv_sh(jobenv)
        jobenv = savejobenv_json(jobenv)
        jobenv = savejobenv_yaml(jobenv)
    for key, val in jobenv.items():
        os.environ[key] = f"{val}"
    if print_jobenv:
        log.info(f"jobenv={json.dumps(jobenv, indent=4, sort_keys=True)}")
    if verbose:
        log.critical(
            "\n".join(
                [
                    "",
                    "Run:",
                    "",
                    f"    source {dotenv_file.relative_to(os.getcwd()).as_posix()}",
                    "",
                    "to set these environment variables.",
                    "",
                ]
            )
        )


def get_jobdirs_from_jobslog() -> list[str]:
    jobslog_file = get_jobslog_file()
    jobdirs: list[str] = []
    if jobslog_file.is_file():
        with jobslog_file.open("r") as f:
            jobdirs.extend([jd.rstrip("\n") for jd in f.readlines()])
    return jobdirs


def get_jobdir_from_jobslog(idx: int = -1) -> str:
    # return Path(jobdirs[0] if len(jobdirs) == 1 else jobdirs[-idx]
    # jobdirs = get_jobdirs_from_jobslog()
    # if len(jobdirs) > 0:
    #     jobdir = jobdirs[0] if len(jobdirs) == 1 else jobdirs[-idx]
    # else:
    #     jobdir = get_jobdir_from_env()
    # return Path(jobdir).as_posix()
    return get_jobdir_from_env().as_posix()


def loadjobenv_from_yaml(
    jobdir: Optional[str | Path] = None,  # type:ignore[reportDeprecated]
) -> dict[str, str]:
    jobdir = Path(get_jobdir_from_jobslog(-1) if jobdir is None else jobdir)
    assert jobdir.is_dir()
    if len((jobenv_files_yaml := list(jobdir.rglob("*.yaml")))) == 0:
        raise FileNotFoundError(
            f"Unable to find `.yaml` file(s) in `{jobdir=}`"
        )
    jobenv_file = jobenv_files_yaml[0]
    with jobenv_file.open("r") as stream:
        jobenv = dict(yaml.safe_load(stream))
    return jobenv


def loadjobenv(jobdir: Optional[str | Path] = None) -> dict[str, str]:
    from ezpz.dist import get_pbs_launch_info, get_dist_info

    jobenv = {}
    jobdir = Path(get_jobdir_from_jobslog(-1) if jobdir is None else jobdir)
    assert jobdir.is_dir()
    jobenv = loadjobenv_from_yaml(jobdir=jobdir)
    jobenv |= get_pbs_launch_info()
    jobenv |= {
        f"{k.upper()}": f"{v}"
        for k, v in (get_dist_info("pytorch", verbose=False).items())
    }
    for key, val in jobenv.items():
        os.environ[key] = val.as_posix() if isinstance(val, Path) else f"{val}"
    dotenv_file = save_to_dotenv_file(jobenv)
    # print_json(data=jobenv, indent=4, sort_keys=True)
    log.info(f"jobenv={json.dumps(jobenv, indent=4, sort_keys=True)}")
    log.critical(
        "\n".join(
            [
                "",
                "Run:",
                "",
                f"    source {dotenv_file.as_posix()}",
                "",
                "to set these environment variables.",
                "",
            ]
        )
    )
    return jobenv


def main(
    hostfile: Optional[str] = None,
    # max_hosts_to_print: Optional[int] = None,
):
    # scheduler = get_scheduler()
    # from ezpz.dist import get_dist_info
    # dinfo = get_dist_info(
    #     hostfile=hostfile,
    #     max_hosts_to_print=max_hosts_to_print,
    # )
    # if scheduler.lower() == 'pbs':
    #     from ezpz.dist import get_pbs_launch_cmd
    #     dinfo |= {'LAUNCH_CMD': get_pbs_launch_cmd(hostfile=hostfile)}
    # log.info(
    #     '\n'.join(
    #         ['\n', "[DIST_INFO]:"]
    #         + [f"  • {k}={v}" for k, v in dinfo.items()]
    #         + ['\n']
    #     )
    # )
    # line = None
    # last_jobdir = None
    # jobenv_file_sh = None
    # if hostfile is None:
    #     PBS_JOBID = os.environ.get('PBS_JOBID')
    #     pbsnf = Path(os.environ.get('PBS_NODEFILE', ''))
    #     if (PBS_JOBID is not None and pbsnf.is_file()):
    #         log.info(f'Caught {PBS_JOBID=}, {pbsnf=} from env. Saving jobenv!')
    #         savejobenv(verbose=False)
    #     else:
    #         log.info('Didnt catch PBS_JOBID in env, loading jobenv!')
    #         _ = loadjobenv()
    # else:
    # _ = get_launch_cmd(verbose=True)
    try:
        savejobenv(
            framework="pytorch",
            hostfile=hostfile,
            # max_hosts_to_print=max_hosts_to_print,
            verbose=True,
            verbose_dist_info=True,
            print_jobenv=False,
            verbose_dotenv=False,
            verbose_get_jobenv=False,
        )
    except Exception as exc:
        log.exception(exc)
        # log.info('Didnt catch PBS_JOBID in env, loading jobenv!')
    # finally:
    #     _ = loadjobenv()


if __name__ == "__main__":
    # import sys
    # import rich
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostfile",
        required=False,
        help=" ".join(
            [
                "Path to hostfile to use.",
                "If not specified, will use $PBS_NODEFILE from environment.",
            ]
        ),
    )
    args = parser.parse_args()
    main(hostfile=args.hostfile)
    # args = sys.argv[1:]
    # # print()
    # if len(args) == 1:
    #     main(hostfile=args[1])
    # elif len(args) == 2:
    #     main(
    #         hostfile=args[1],
    #         max_hosts_to_print=int(args[2])
    #     )
    # else:
    #     main()
    # scheduler = get_scheduler()
    # line = None
    # last_jobdir = None
    # jobenv_file_sh = None
    # PBS_JOBID = os.environ.get('PBS_JOBID')
    # pbsnf = Path(os.environ.get('PBS_NODEFILE', ''))
    # if (PBS_JOBID is not None and pbsnf.is_file()):
    #     log.info(f'Caught {PBS_JOBID=}, {pbsnf=} from env. Saving jobenv!')
    #     savejobenv(verbose=False)
    # else:
    #     log.info('Didnt catch PBS_JOBID in env, loading jobenv!')
    #     _ = loadjobenv()
    #
    # from ezpz.dist import get_dist_info
    # dinfo = get_dist_info()
    # log.info(
    #     '\n'.join(
    #         ['\n', "[DIST_INFO]:"]
    #         + [f"  • {k}={v}" for k, v in dinfo.items()]
    #     )
    # )
    # _ = get_launch_cmd(verbose=True)
