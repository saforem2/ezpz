"""
jobs.py
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import logging.config
import os
import json
import yaml
from pathlib import Path
from typing import Optional

from ezpz import (
    get_dist_info,
    get_gpus_per_node,
    get_hosts_from_hostfile,
    get_machine,
    get_torch_backend,
    get_torch_device,
)
from ezpz.configs import get_logging_config, get_scheduler, SCHEDULERS

log_config = logging.config.dictConfig(get_logging_config())
log = logging.getLogger(__name__)

log.setLevel('INFO')

SCHEDULER = get_scheduler()


def get_pbs_env(verbose: bool = False) -> dict[str, str]:
    pbsenv = {
        k: v for k, v in dict(os.environ).items() if 'PBS' in k
    }
    hostfile = pbsenv.get('PBS_NODEFILE')
    if hostfile is not None and (hfp := Path(hostfile)).is_file():
        HOSTFILE, hosts = get_hosts_from_hostfile(hfp)
        hosts = [h.split('.')[0] for h in hosts]
        nhosts = len(hosts)
        ngpu_per_host = get_gpus_per_node()
        ngpus = nhosts * ngpu_per_host
        launch_cmd = ' '.join([
            'mpiexec',
            '--verbose',
            '--envall',
            f'-n {ngpus}',
            f'-ppn {ngpu_per_host}',
            f'--hostfile {HOSTFILE}'
        ])
        launch_info = {
            'HOSTFILE': HOSTFILE,
            'HOSTS': f'[{", ".join(hosts)}]',
            'NHOSTS': f'{nhosts}',
            'NGPU_PER_HOST': f'{ngpu_per_host}',
            'NGPUS': f'{ngpus}',
            'machine': get_machine(),
            'device': get_torch_device(),
            'backend': get_torch_backend(),
            'launch_cmd': launch_cmd,
        }
        os.environ |= launch_info
        pbsenv |= launch_info
    dist_info = get_dist_info(framework='pytorch', verbose=verbose)
    # dist_info.pop('')
    pbsenv |= {k: f'{v}' for k, v in dist_info.items()}
    return pbsenv


def check_scheduler(scheduler: Optional[str] = None) -> bool:
    scheduler = SCHEDULER if scheduler is None else scheduler
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
    jobid = get_jobid()
    jobdir = get_jobdir_from_env()
    return jobdir.joinpath(f'{SCHEDULER}-{jobid}.{ext}')


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


def get_jobenv() -> dict:
    if SCHEDULER.lower() == 'pbs':
        return get_pbs_env()
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
            f.write(f'{jobdir}\n')
    else:
        log.warning(
            f'{jobdir.as_posix()} '
            f'already in {jobslog_file.as_posix()}, '
            f'not appending !!'
        )


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
        f.write(contents)
    os.chmod(path=launch_file, mode=755)
    path = os.environ.get('PATH')
    path = f'{path}:$HOME/.local/bin'
    os.environ['PATH'] = f'{path}'


def savejobenv_sh(jobenv: Optional[dict] = None) -> dict:
    jobenv = get_jobenv() if jobenv is None else jobenv
    jobfile_sh = get_jobfile_sh()
    jobenv |= {'jobfile_sh': jobfile_sh.as_posix()}
    log.info(f'Saving job env to {jobfile_sh}')
    with jobfile_sh.open('w') as f:
        f.write('#!/bin/bash --login\n')
        for key, val in jobenv.items():
            f.write(f'export {key.upper()}="{val}"\n')
        launch_cmd = jobenv.get('launch_cmd')
        if launch_cmd is not None:
            f.write(f'alias launch="{launch_cmd}"')
    dotenv_file = Path(os.getcwd()).joinpath('.env')
    log.info(
        f'Saving job env to dot-env (.env) file in '
        f'{dotenv_file.parent.as_posix()}'
    )
    with dotenv_file.open('w') as f:
        for key, val in jobenv.items():
            f.write(f'{key.upper()}="{val}"\n')
    return jobenv


def savejobenv_json(jobenv: Optional[dict] = None) -> dict:
    jobenv = get_jobenv() if jobenv is None else jobenv
    assert len(jobenv.keys()) > 0
    jobfile_json = get_jobfile_json()
    jobenv |= {'jobfile_json': jobfile_json.as_posix()}
    log.info(f'Saving job env to {jobfile_json}')
    with jobfile_json.open('w') as f:
        json.dump(json.dumps(jobenv, indent=4), f)
    return jobenv


def savejobenv_yaml(
        jobenv: Optional[dict] = None,
        # scheduler: Optional[str] = None
) -> dict:
    jobenv = get_jobenv() if jobenv is None else jobenv
    assert len(jobenv.keys()) > 0
    jobfile_yaml = get_jobfile_yaml()
    jobenv |= {'jobfile_yaml': jobfile_yaml.as_posix()}
    log.info(f'Saving job env to {jobfile_yaml}')
    with jobfile_yaml.open('w') as f:
        yaml.dump(jobenv, f)
    return jobenv


def savejobenv():
    jobenv = get_jobenv()
    assert len(jobenv.keys()) > 0
    jobid = get_jobid()
    jobdir = get_jobdir_from_env()
    assert jobenv is not None
    assert jobdir is not None
    from rich import print_json
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
    for key, val in jobenv.items():
        os.environ[key] = f'{val}'
    log.info(
        f'Writing {SCHEDULER} env vars to '
        f'{jobdir} / {SCHEDULER}-{jobid}' + '{.sh, .yaml, .json}'
    )
    print_json(data=jobenv, indent=4, sort_keys=True)
    # ---------------------------------------------------


def get_jobdirs_from_jobslog() -> list[str]:
    jobslog_file = get_jobslog_file()
    jobdirs = []
    if jobslog_file.is_file():
        with jobslog_file.open('r') as f:
            jobdirs.extend([jd.rstrip('\n') for jd in f.readlines()])
    return jobdirs


def get_jobdir_from_jobslog(
        idx: int = -1,
) -> str:
    jobdirs = get_jobdirs_from_jobslog()
    return jobdirs[0] if len(jobdirs) == 1 else jobdirs[-idx]


def loadjobenv() -> dict:
    from rich import print_json
    jobenv = {}
    last_jobdir = Path(get_jobdir_from_jobslog(-1))
    assert last_jobdir.is_dir()
    import yaml
    if len((jobenv_files_yaml := list(last_jobdir.rglob('*.yaml')))) > 0:
        jobenv_file = jobenv_files_yaml[0]
        with jobenv_file.open('r') as stream:
            jobenv = dict(yaml.safe_load(stream))
        for key, val in jobenv.items():
            os.environ[key] = val
    print_json(data=jobenv, indent=4, sort_keys=True)
    return jobenv


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    scheduler = get_scheduler()
    line = None
    last_jobdir = None
    jobenv_file_sh = None
    if args[0].lower().startswith('save'):
        savejobenv()
    elif args[0].lower().startswith('get'):
        loadjobenv()
