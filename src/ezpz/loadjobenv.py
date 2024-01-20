"""
ezpz/getjob.py

Wrapper script around the `loadjobenv()` function from `jobs.py`
"""


if __name__ == '__main__':
    from ezpz.jobs import loadjobenv
    jobenv = loadjobenv()
