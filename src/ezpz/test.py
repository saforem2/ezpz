from typing import Optional
from ezpz import setup_torch


def main(backend: Optional[str] = None):
    backend = 'DDP' if backend is None else backend
    RANK = setup_torch(backend=backend)
    print(RANK)


if __name__ == '__main__':
    import sys
    backend = None
    if len(sys.argv) > 1:
        backend = sys.argv[1]
        assert isinstance(backend, str) and backend.lower() in ['ddp', 'deepspeed', 'horovod']
    main(backend)
