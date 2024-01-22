"""
ezpz/loadjobenv.py

Wrapper script around the `loadjobenv()` function from `jobs.py`
"""


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    from ezpz.jobs import loadjobenv
    jobenv = loadjobenv()
