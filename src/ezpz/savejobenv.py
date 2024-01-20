"""
ezpz/savejob.py

Wrapper script around the `savejobenv()` function from `jobs.py`
"""


if __name__ == '__main__':
    # import sys
    # args = sys.argv[1:]
    # scheduler = get_scheduler()
    # if args[0].lower().startswith('save'):
    from ezpz.jobs import savejobenv
    savejobenv()
    #save_job(scheduler=scheduler)

