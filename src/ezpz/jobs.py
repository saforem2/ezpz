"""
jobs.py
"""
from __future__ import absolute_import, annotations, division, print_function

import os
from pathlib import Path
from typing import Optional

SCHEDULERS = {
    'PBS',
    'SLURM',
    'COBALT',
}


def get_pbs_env() -> dict[str, str]:
    return {
        k: v for k, v in dict(os.environ).items() if 'PBS' in k
    }

def get_jobdir_from_env(scheduler: str = 'pbs') -> Path:
    assert scheduler in SCHEDULERS
    if scheduler == 'pbs':
        pbs_env = get_pbs_env()
        jobid = pbs_env["PBS_JOBID"].split('.')[0]
    else:
        raise TypeError(f'{scheduler} not yet implemented!')
    jobdir = Path.home() / f'{scheduler}-jobs' / f'{jobid}' 
    jobdir.parent.mkdir(exist_ok=True, parents=True)
    return jobdir


def get_jobid(scheduler: str = 'pbs') -> str:
    if scheduler.lower() == 'pbs':
        pbs_env = get_pbs_env()
    else:
        raise ValueError(f'{scheduler} not implemented')
    jobid = pbs_env['PBS_JOBID'].split('.')[0]
    return jobid


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
    assert scheduler in SCHEDULERS
    if scheduler.lower() == 'pbs':
        return get_pbs_env()
    raise ValueError(f'{scheduler} not yet implemented!')


def get_jobslog_file(scheduler: str = 'pbs') -> Path:
    return Path.home().joinpath(f'{scheduler}-jobs.log')


def add_to_jobslog(scheduler: str = 'pbs'):
    jobenv = get_jobenv(scheduler=scheduler)
    jobdir = get_jobdir_from_env(scheduler=scheduler)
    assert jobenv is not None
    assert jobdir is not None
    jobslog_file = get_jobslog_file(scheduler=scheduler)
    with jobslog_file.open('a') as f:
        f.write(f'{jobdir}\n')


def savejobenv_sh(jobenv: dict, scheduler: str = 'pbs'):
    jobfile_sh = get_jobfile_sh(scheduler=scheduler)
    print(f'Saving job env to {jobfile_sh}')
    with jobfile_sh.open('w') as f:
        for key, val in jobenv.items():
            f.write(f'export {key.upper()}={val}\n')

def savejobenv_json(jobenv: dict, scheduler: str = 'pbs'):
    import json
    jobfile_json = get_jobfile_json(scheduler=scheduler)
    print(f'Saving job env to {jobfile_json}')
    with jobfile_json.open('w') as f:
        json.dump(json.dumps(jobenv, indent=4), f)


def savejobenv_yaml(jobenv: dict, scheduler: str = 'pbs'):
    import yaml
    jobfile_yaml = get_jobfile_yaml(scheduler=scheduler)
    print(f'Saving job env to {jobfile_yaml}')
    with jobfile_yaml.open('w') as f:
        yaml.dump(jobenv, f)


def savejobenv(jobenv: dict, scheduler: str = 'pbs'):
    jobid = get_jobid(scheduler)
    jobenv = get_jobenv(scheduler=scheduler)
    jobdir = get_jobdir_from_env(scheduler=scheduler)
    assert jobenv is not None
    assert jobdir is not None
    print(
        f'Writing {scheduler} env vars to '
        f'{jobdir} / {scheduler}-{jobid}.' + '{sh,yaml,json}'
    )
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


def get_jobdir_from_jobslog(
        idx: int = -1,
        jobid: Optional[int | str] = None,
        scheduler: str = 'pbs',
):
    # jobdir = get_jobdir_from_env(scheduler=scheduler)
    jobslog_file = get_jobslog_file(scheduler=scheduler)
    lines = []
    # with jobslog_file.open('r') as f:
    #     for line in f.readlines()[-idx]
    pass

    # for line in jobslog_file.open('r') as f:
    #     for line in file.readlines()[-]
    #         for line in (file.readlines() [-N:]):
    #         print(line, end ='')


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if args[0].lower() == 'savejobenv':
        save_job()
    elif args[0].lower() == 'getjobenv':
        pass
