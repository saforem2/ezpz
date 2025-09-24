"""
ezpz/savejobenv.py

Wrapper script around the `savejobenv()` function from `jobs.py`
"""


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    from ezpz.jobs import savejobenv

    savejobenv()
