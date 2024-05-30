"""
jobs.py
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
import json
import yaml
from pathlib import Path
from typing import Optional, Any
# from rich import print_json

from ezpz import (
    get_dist_info,
)
from ezpz.dist import (
    get_pbs_env,
    get_pbs_launch_info,
)
from ezpz.configs import (
    get_logging_config,
    get_scheduler,
    SCHEDULERS,
    PathLike
)

# log_config = logging.config.dictConfig(get_logging_config())
log = logging.getLogger(__name__)
log.setLevel('INFO')

SCHEDULER = get_scheduler()


def check_scheduler(scheduler: Optional[str] = None) -> bool:
    scheduler = SCHEDULER if scheduler is None else scheduler
    if scheduler is not None and len(scheduler) > 0:
        assert scheduler.upper() in SCHEDULERS.values()
    if scheduler.lower() != 'pbs':
        raise TypeError(f'{scheduler} not yet implemented!')
    return True


# def get_jobdir_from_env(scheduler: Optional[str] = None) -> Path:
def get_jobdir_from_env() -> Path:
    pbs_env = get_pbs_env()
    jobid = pbs_env["PBS_JOBID"].split('.')[0]
    jobdir = Path.home() / f'{SCHEDULER}-jobs' / f'{jobid}'
    jobdir.mkdir(exist_ok=True, parents=True)
    return jobdir


def get_jobid() -> str:
    jobenv = get_jobenv()
    return jobenv['PBS_JOBID'].split('.')[0]


def get_jobfile_ext(ext: str) -> Path:
    # jobid = get_jobid()
    jobdir = get_jobdir_from_env()
    # return jobdir.joinpath(f'{SCHEDULER}-{jobid}.{ext}')
    return jobdir.joinpath(f'jobenv.{ext}')


def get_jobfile_sh() -> Path:
    jobfile_sh = get_jobfile_ext('sh')
    jobfile_sh.parent.mkdir(exist_ok=True, parents=True)
    return jobfile_sh


def get_jobfile_yaml() -> Path:
    jobfile_ext = get_jobfile_ext('yaml')
    jobfile_ext.parent.mkdir(exist_ok=True, parents=True)
    return jobfile_ext


def get_jobfile_json() -> Path:
    jobfile_ext = get_jobfile_ext('json')
    jobfile_ext.parent.mkdir(exist_ok=True, parents=True)
    return jobfile_ext


def get_jobenv(verbose: bool = False) -> dict:
    from ezpz.dist import get_pbs_launch_info
    jobenv: dict[str, str | int | list[Any]] = get_dist_info(framework='pytorch', verbose=verbose)
    if SCHEDULER.lower() == 'pbs':
        jobenv |= get_pbs_env()
        jobenv |= get_pbs_launch_info()
        jobenv |= get_dist_info()
        return jobenv
    # TODO: Add Slurm support to Python API
    raise ValueError(f'{SCHEDULER} not yet implemented!')


def get_jobslog_file() -> Path:
    jobslog_file = Path.home().joinpath(f'{SCHEDULER}-jobs.log')
    jobslog_file.parent.mkdir(exist_ok=True, parents=True)
    return jobslog_file


def add_to_jobslog():
    jobenv = get_jobenv()
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
        with jobslog_file.open('a') as f:
            _ = f.write(f'{jobdir}\n')
    else:
        log.warning(' '.join([
            f'{jobdir.as_posix()} ',
            f'already in {jobslog_file.as_posix()}, ',
            f'not appending !!',
        ]))


def save_to_dotenv_file(
        jobenv: Optional[dict[str, str]] = None,  # type:ignore[reportDeprecated]
) -> Path:
    jobenv = get_jobenv() if jobenv is None else jobenv
    if (
            'LAUNCH_CMD' in jobenv and 'launch_cmd' in jobenv
            and jobenv['launch_cmd'] != jobenv['LAUNCH_CMD']
    ):
        jobenv['launch_cmd'] = jobenv['LAUNCH_CMD']
        jobenv['DIST_LAUNCH'] = jobenv['LAUNCH_CMD']
    denvf1 = Path(get_jobdir_from_env()).joinpath('.jobenv')
    denvf2 = Path(os.getcwd()).joinpath('.jobenv')
    for denvf in [denvf1, denvf2]:
        log.info(' '.join([
            f'Saving job env to `.jobenv` file in ',
            f'{denvf.parent.as_posix()}/.jobenv'
        ]))
        launch_cmd = jobenv.get('LAUNCH_CMD')
        assert launch_cmd is not None
        # launch_cmd = jobenv.get(
        #         'LAUNCH_CMD',
        #         get_jobenv().get('LAUNCH_CMD', '')
        # )
        # if launch_cmd is not None:
        with denvf.open('w') as f:
            _ = f.write('#!/bin/bash --login\n')
            for key, val in jobenv.items():
                _ = f.write(f'{key.upper()}="{val}"\n')
            _ = f.write(f'echo "creating alias launch={launch_cmd}"\n')
            _ = f.write(f'alias launch="{launch_cmd}"\n')
    log.warning(' '.join([
        f'To use `launch` alias, be sure to: ',
        f'`source {denvf2.as_posix()}'
    ]))
    return denvf2


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
    local_bin = Path().home().joinpath('.local', 'bin')
    local_bin.mkdir(exist_ok=True, parents=True)
    launch_file = local_bin.joinpath('launch.sh')
    # launch_file.chmod(launch_file.stat().st_mode | stat.S_IEXEC)
    log.info(f'Saving launch command to {launch_file} and adding to PATH')
    with launch_file.open('w') as f:
        _ = f.write(contents)
    os.chmod(path=launch_file, mode=755)
    path = os.environ.get('PATH')
    path = f'{path}:$HOME/.local/bin'
    os.environ['PATH'] = f'{path}'


def savejobenv_sh(
        jobenv: Optional[dict[str, str]] = None  # type:ignore[reportDeprecated]
) -> dict[str, str]:
    jobenv = get_jobenv() if jobenv is None else jobenv
    jobfile_sh = get_jobfile_sh()
    jobenv |= {'jobfile_sh': jobfile_sh.as_posix()}
    launch_cmd = jobenv.get('LAUNCH_CMD')
    log.info(f'Saving job env to {jobfile_sh}')
    with jobfile_sh.open('w') as f:
        _ = f.write('#!/bin/bash --login\n')
        for key, val in jobenv.items():
            _ = f.write(f'export {key.upper()}="{val}"\n')
        if launch_cmd is not None:
            _ = f.write(f'alias launch="{launch_cmd}"')
    # dotenv_file = Path(os.getcwd()).joinpath('.jobenv')
    # log.info(' '.join([
    #     f'Saving job env to dot-env (`.jobenv`) file in ',
    #     f'{dotenv_file.parent.as_posix()}'
    # ]))
    # with dotenv_file.open('w') as f:
    #     _ = f.write('#!/bin/bash --login\n')
    #     for key, val in jobenv.items():
    #         _ = f.write(f'{key.upper()}="{val}"\n')
    #     if launch_cmd is not None:
    #         _ = f.write(f'alias launch="{launch_cmd}"')
    return jobenv


def savejobenv_json(
        jobenv: Optional[dict[str, str]] = None  # type:ignore[reportDeprecated]
) -> dict[str, str]:
    jobenv = get_jobenv() if jobenv is None else jobenv
    assert len(jobenv.keys()) > 0
    jobfile_json = get_jobfile_json()
    jobenv |= {'jobfile_json': jobfile_json.as_posix()}
    log.info(f'Saving job env to {jobfile_json}')
    with jobfile_json.open('w') as f:
        json.dump(json.dumps(jobenv, indent=4), f)
    return jobenv


def savejobenv_yaml(
    jobenv: Optional[dict[str, str]] = None,  # type:ignore[reportDeprecated]
) -> dict[str, str]:
    jobenv = get_jobenv() if jobenv is None else jobenv
    assert len(jobenv.keys()) > 0
    jobfile_yaml = get_jobfile_yaml()
    jobenv |= {'jobfile_yaml': jobfile_yaml.as_posix()}
    log.info(f'Saving job env to {jobfile_yaml}')
    with jobfile_yaml.open('w') as f:
        yaml.dump(jobenv, f)
    return jobenv


def savejobenv():
    jobenv: dict[str, Any] = get_jobenv()
    assert len(jobenv.keys()) > 0
    # jobid = get_jobid()
    jobdir = get_jobdir_from_env()
    assert jobenv is not None
    assert jobdir is not None
    # from rich import print_json
    # -------------------------------------------------------------------
    # Append {jobdir} as a new line at the end of ~/{scheduler}-jobs.log
    # where:
    #   jobdir = Path.home() / f'{scheduler}-jobs' / f'{jobid}'
    add_to_jobslog()
    # -------------------------------------------------------------------
    # Save {scheduler}-related environment variables to
    # `{.sh,.yaml,.json}` files INSIDE {jobdir}
    # for easy loading in other processes
    jobenv = savejobenv_sh(jobenv)
    jobenv = savejobenv_json(jobenv)
    jobenv = savejobenv_yaml(jobenv)
    _ = save_to_dotenv_file(jobenv)
    for key, val in jobenv.items():
        os.environ[key] = f'{val}'
    log.info(f'jobenv={json.dumps(jobenv, indent=4, sort_keys=True)}')
    log.info(' '.join([
        f'Writing {SCHEDULER} env vars to ',
        f'{jobdir} / jobenv' + '{.sh, .yaml, .json}'
    ]))
    log.critical(
        f'Run: `source ./.jobenv` in your current shell to set job variables'
    )
    # ---------------------------------------------------


def get_jobdirs_from_jobslog() -> list[str]:
    jobslog_file = get_jobslog_file()
    jobdirs: list[str] = []
    if jobslog_file.is_file():
        with jobslog_file.open('r') as f:
            jobdirs.extend([jd.rstrip('\n') for jd in f.readlines()])
    return jobdirs


def get_jobdir_from_jobslog(
        idx: int = -1,
) -> str:
    # return Path(jobdirs[0] if len(jobdirs) == 1 else jobdirs[-idx]
    # jobdirs = get_jobdirs_from_jobslog()
    # if len(jobdirs) > 0:
    #     jobdir = jobdirs[0] if len(jobdirs) == 1 else jobdirs[-idx]
    # else:
    #     jobdir = get_jobdir_from_env()
    # return Path(jobdir).as_posix()
    return get_jobdir_from_env().as_posix()


def loadjobenv_from_yaml(
        jobdir: Optional[str | Path] = None  # type:ignore[reportDeprecated]
) -> dict[str, str]:
    jobdir = Path(get_jobdir_from_jobslog(-1) if jobdir is None else jobdir)
    assert jobdir.is_dir()
    if len((jobenv_files_yaml := list(jobdir.rglob('*.yaml')))) == 0:
        raise FileNotFoundError(
            f'Unable to find `.yaml` file(s) in `{jobdir=}`'
        )
    jobenv_file = jobenv_files_yaml[0]
    with jobenv_file.open('r') as stream:
        jobenv = dict(yaml.safe_load(stream))
    return jobenv


def loadjobenv(jobdir: Optional[str | Path] = None) -> dict[str, str]:
    jobenv = {}
    jobdir = Path(
        get_jobdir_from_jobslog(-1) if jobdir is None else jobdir
    )
    assert jobdir.is_dir()
    jobenv = loadjobenv_from_yaml(jobdir=jobdir)
    jobenv |= get_pbs_launch_info()
    jobenv |= {
        f'{k.upper()}': f'{v}' for k, v in (
            get_dist_info('pytorch', verbose=False).items()
        )
    }
    for key, val in jobenv.items():
        os.environ[key] = (
            val.as_posix() if isinstance(val, Path)
            else f'{val}'
        )
    dotenv_file = save_to_dotenv_file(jobenv)
    # print_json(data=jobenv, indent=4, sort_keys=True)
    log.info(f'jobenv={json.dumps(jobenv, indent=4, sort_keys=True)}')
    log.critical(
        '\n'.join(
            [
                'Run: `source ./.jobenvenv` in your CURRENT shell to load these!!',
                f'[Note] full_path: {dotenv_file.as_posix()}'
            ]
        )
    )
    return jobenv


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    scheduler = get_scheduler()
    line = None
    last_jobdir = None
    jobenv_file_sh = None
    PBS_JOBID = os.environ.get('PBS_JOBID')
    pbsnf = Path(os.environ.get('PBS_NODEFILE', ''))
    if (PBS_JOBID is not None and pbsnf.is_file()):
        log.info(f'Caught {PBS_JOBID=}, {pbsnf=} from env. Saving jobenv!')
        savejobenv()
    else:
        log.info('Didnt catch PBS_JOBID in env, loading jobenv!')
        _ = loadjobenv()
