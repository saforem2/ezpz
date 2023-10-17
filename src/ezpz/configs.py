"""
ezpz/configs.py
"""
from typing import Optional
from dataclasses import dataclass


FRAMEWORKS = {
    'pytorch': ['p', 'pt', 'torch', 'pytorch'],
    'tensorflow': ['t', 'tf', 'tflow', 'tensorflow'],
}
BACKENDS = {
    'pytorch': ['ddp', 'deepspeed', 'ds', 'horovod', 'hvd'],
    'tensorflow': ['hvd', 'horovod']
}


@dataclass
class TrainConfig:
    framework: str = "pytorch"
    backend: str = "DDP"
    seed: Optional[int] = None
    port: Optional[str] = None

    def __post_init__(self):
        assert self.framework in [
            'p',
            'pt',
            'pytorch',
            't',
            'tf',
            'tensorflow',
        ]
        if self.framework in ["t", "tf" ,"tensorflow"]:
            assert self.backend in ['h', 'hvd', 'horovod']
