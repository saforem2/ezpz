"""
runtime.py

Contains Python functions to be used at runtime.
"""
import pdb
import sys

import tqdm


class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in:
    https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """

    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        # if len(x.rstrip()) > 0:
        tqdm.tqdm.write(x, file=self.file, end="\n")

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()
