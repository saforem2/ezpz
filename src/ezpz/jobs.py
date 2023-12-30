"""
jobs.py
"""
from __future__ import absolute_import, annotations, division, print_function

import os
from pathlib import Path

SCHEDULERS = {
    'PBS',
    'SLURM',
    'COBALT',
}


def get_pbs_env() -> dict[str, str]:
    return {
        k: v for k, v in dict(os.environ).items() if 'PBS' in k
    }


def get_jobfile(scheduler: str = 'pbs') -> Path:
    assert scheduler in SCHEDULERS
    if scheduler == 'pbs':
        pbs_env = get_pbs_env()
        jobid = pbs_env["PBS_JOBID"].split('.')[0]
    else:
        raise TypeError(f'{scheduler} not yet implemented!')
    jobfile = Path.home() / f'{scheduler}jobs' / f'{jobid}' / f'pbs-{jobid}.sh'
    jobfile.parent.mkdir(exist_ok=True, parents=True)
    return jobfile


def write_to_jobsfile(scheduler: str = 'pbs'):
    assert scheduler in SCHEDULERS
    if scheduler == 'pbs':
        # pbs_env = get_pbs_env()
        jobfile = get_jobfile()
        # jobid = pbs_env["PBS_JOBID"].split('.')[0]
    else:
        raise TypeError(f'{scheduler} not yet implemented!')
    jobsfile = Path.home().joinpath(f'{scheduler}-jobs.log')
    print(f'Adding {jobfile.as_posix()} to {jobsfile.as_posix()}')
    with jobsfile.open('w') as f:
        f.write(f'{jobfile}\n')


def save_job():
    pbs_env = get_pbs_env()
    jobfile = get_jobfile()
    assert len(pbs_env.keys()) > 0, (
        'No `PBS_*` variables in environment...'
    )
    with jobfile.open('w') as f:
        for key, val in pbs_env.items():
            f.write(f'export {key}={val}\n')


def 

if __name__ == '__main__':
