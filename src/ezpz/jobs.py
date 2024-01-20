"""
jobs.py
"""
from __future__ import absolute_import, annotations, division, print_function
# import logging
# import logging.config
import os
from pathlib import Path
# from typing import Optional

from rich.text import Text

# from ezpz.configs import get_logging_config
from ezpz.dist import (
    get_gpus_per_node,
    get_hosts_from_hostfile,
    get_machine,
    get_rank,
    get_dist_info,
    get_hostname,
    get_torch_backend,
    get_torch_device,
)

# logging.config.dictConfig(get_logging_config())
# log = logging.getLogger(__name__)
from enrich.console import get_console

console = get_console()


SCHEDULERS = {
    'PBS',
    'SLURM',
    'COBALT',
}


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
        # export LAUNCH_CMD="mpiexec --verbose --envall -n ${NGPUS}
        # -ppn $NGPU_PER_HOST --hostfile ${HOSTFILE}"   # "$@"
        launch_cmd = ' '.join([
            'mpiexec',
            '--verbose',
            '--envall',
            f'-n {ngpus}',
            f'-ppn {ngpu_per_host}',
            f'--hostfile {HOSTFILE}'
        ])
        # pbsenv |= {'HOSTFILE': HOSTFILE, 'hosts': hosts}
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
        # os.environ.update(launch_info)
        # pbsenv.update(launch_info)
    pbsenv |= get_dist_info(framework='pytorch', verbose=verbose)
    return pbsenv


def get_jobdir_from_env(scheduler: str = 'pbs') -> Path:
    assert scheduler.upper() in SCHEDULERS
    if scheduler.lower() != 'pbs':
        raise TypeError(f'{scheduler} not yet implemented!')
    pbs_env = get_pbs_env()
    jobid = pbs_env["PBS_JOBID"].split('.')[0]
    jobdir = Path.home() / f'{scheduler}-jobs' / f'{jobid}'
    jobdir.mkdir(exist_ok=True, parents=True)
    # jobdir.parent.mkdir(exist_ok=True, parents=True)
    return jobdir


def get_jobid(scheduler: str = 'pbs') -> str:
    if scheduler.lower() == 'pbs':
        pbs_env = get_pbs_env()
    else:
        raise ValueError(f'{scheduler} not implemented')
    return pbs_env['PBS_JOBID'].split('.')[0]


def get_jobfile_ext(ext: str, scheduler: str = 'pbs') -> Path:
    jobid = get_jobid(scheduler)
    jobdir = get_jobdir_from_env(scheduler)
    return jobdir.joinpath(f'{scheduler}-{jobid}.{ext}')


def get_jobfile_sh(scheduler: str = 'pbs') -> Path:
    return get_jobfile_ext('sh', scheduler=scheduler)


def get_jobfile_yaml(scheduler: str = 'pbs') -> Path:
    # jobid = get_jobid(scheduler)
    # jobdir = get_jobdir_from_env(scheduler)
    # return jobdir.joinpath(f'{scheduler}-{jobid}.yaml')
    return get_jobfile_ext('yaml', scheduler=scheduler)


def get_jobfile_json(scheduler: str = 'pbs') -> Path:
    return get_jobfile_ext('json', scheduler=scheduler)


def get_jobenv(scheduler: str = 'pbs') -> dict:
    assert scheduler.upper() in SCHEDULERS
    if scheduler.lower() == 'pbs':
        return get_pbs_env()
    raise ValueError(f'{scheduler} not yet implemented!')


def get_jobslog_file(scheduler: str = 'pbs') -> Path:
    return Path.home().joinpath(f'{scheduler}-jobs.log')


def add_to_jobslog(scheduler: str = 'pbs'):
    jobenv = get_jobenv(scheduler=scheduler)
    assert len(jobenv.keys()) > 0
    jobdir = get_jobdir_from_env(scheduler=scheduler)
    assert jobenv is not None
    assert jobdir is not None
    jobslog_file = get_jobslog_file(scheduler=scheduler)
    last_jobdir = get_jobdir_from_jobslog(-1, scheduler=scheduler)
    if jobdir.as_posix() != last_jobdir:
        with jobslog_file.open('a') as f:
            f.write(f'{jobdir}\n')
    else:
        console.print(
            Text(f'[orange]Warning![/orange]'
                 f'{jobdir.as_posix()} already in {jobslog_file.as_posix()}')
        )


def savejobenv_sh(jobenv: dict, scheduler: str = 'pbs'):
    assert len(jobenv.keys()) > 0
    jobfile_sh = get_jobfile_sh(scheduler=scheduler)
    console.print(f'Saving job env to {jobfile_sh}')
    # jobfile_sh.touch(exist_ok=True)
    with jobfile_sh.open('w') as f:
        f.write('#!/bin/bash --login\n')
        for key, val in jobenv.items():
            f.write(f'export {key.upper()}="{val}"\n')
        launch_cmd = jobenv.get('launch_cmd')
        if launch_cmd is not None:
            f.write(f'alias launch="{launch_cmd}"')


def savejobenv_json(jobenv: dict, scheduler: str = 'pbs'):
    assert len(jobenv.keys()) > 0
    import json
    jobfile_json = get_jobfile_json(scheduler=scheduler)
    # jobfile_json.touch(exist_ok=True)
    console.print(f'Saving job env to {jobfile_json}')
    with jobfile_json.open('w') as f:
        json.dump(json.dumps(jobenv, indent=4), f)


def savejobenv_yaml(jobenv: dict, scheduler: str = 'pbs'):
    assert len(jobenv.keys()) > 0
    import yaml
    jobfile_yaml = get_jobfile_yaml(scheduler=scheduler)
    # jobfile_yaml.touch(exist_ok=True)
    console.print(f'Saving job env to {jobfile_yaml}')
    with jobfile_yaml.open('w') as f:
        yaml.dump(jobenv, f)


def savejobenv():
    scheduler = get_scheduler()
    jobenv = get_jobenv(scheduler=scheduler)
    assert len(jobenv.keys()) > 0
    jobid = get_jobid(scheduler=scheduler)
    jobdir = get_jobdir_from_env(scheduler=scheduler)
    assert jobenv is not None
    assert jobdir is not None
    if get_rank() == 0:
        from rich import print_json
        console.print(
            f'Writing {scheduler} env vars to '
            f'{jobdir} / {scheduler}-{jobid}.' + '{sh,yaml,json}'
        )
        print(
            f'Writing {scheduler} env vars to '
            f'{jobdir} / {scheduler}-{jobid}.' + '{sh,yaml,json}'
        )
        print_json(data=jobenv, indent=4)
    # -------------------------------------------------------------------
    # Append {jobdir} as a new line at the end of ~/{scheduler}-jobs.log
    # where:
    #   jobdir = Path.home() / f'{scheduler}-jobs' / f'{jobid}'
    add_to_jobslog(scheduler)
    # -------------------------------------------------------------------
    # Save {scheduler}-related environment variables to
    # `{.sh,.yaml,.json}` files INSIDE {jobdir}
    # for easy loading in other processes
    savejobenv_sh(jobenv, scheduler=scheduler)
    savejobenv_json(jobenv, scheduler=scheduler)
    savejobenv_yaml(jobenv, scheduler=scheduler)
    # ---------------------------------------------------


# def write_to_jobsfile(scheduler: str = 'pbs'):
#     assert scheduler in SCHEDULERS
#     if scheduler == 'pbs':
#         # jobfile_sh = get_jobfile_sh(scheduler)
#         # jobfile_yaml = get_jobfile_yaml(scheduler)
#         # jobfile_json = get_jobfile_json(scheduler)
#         jobid = get_jobid(scheduler)
#         jobdir = get_jobdir_from_env(scheduler=scheduler)
#     else:
#         raise TypeError(f'{scheduler} not yet implemented!')
#     print(
#         f'Writing {scheduler} env vars to '
#         f'{jobdir} / {scheduler}-{jobid}.' + '{sh,yaml,json}'
#     )
#     jobsfile = Path.home().joinpath(f'{scheduler}-jobs.log')
#     print(f'Adding {jobdir.as_posix()} to {jobsfile.as_posix()}')
#     with jobsfile.open('w') as f:
#         f.write(f'{jobsfile}\n')


def save_job(scheduler: str = 'pbs'):
    jobenv = get_jobenv(scheduler=scheduler)
    # jobfile = get_jobfile()
    savejobenv(jobenv, scheduler=scheduler)
    # with jobfile.open('w') as f:
    #     for key, val in pbs_env.items():
    #         f.write(f'export {key}={val}\n')


def get_jobdirs_from_jobslog(
        # idx: int = -1,
        # jobid: Optional[int | str] = None,
        scheduler: str = 'pbs',
) -> list[str]:
    jobslog_file = get_jobslog_file(scheduler=scheduler)
    jobdirs = []
    with jobslog_file.open('r') as f:
        jobdirs.extend([jd.rstrip('\n') for jd in f.readlines()])
    return jobdirs


def get_jobdir_from_jobslog(
        idx: int = -1,
        scheduler: str = 'pbs',
):
    jobdirs = get_jobdirs_from_jobslog(scheduler=scheduler)
    return jobdirs[-idx]


def get_scheduler() -> str:
    from ezpz import get_machine
    machine = get_machine(get_hostname())
    if machine.lower() in ['thetagpu', 'sunspot', 'polaris', 'aurora']:
        return SCHEDULERS['ALCF']
    elif machine.lower() in ['nersc', 'perlmutter']:
        return SCHEDULERS['NERSC']
    raise RuntimeError(f'Unknown {machine=}')


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    scheduler = get_scheduler()
    if args[0].lower() == 'savejobenv':
        save_job(scheduler=scheduler)
    elif args[0].lower() == 'getjobenv':
        pass
