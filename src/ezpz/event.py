"""
event.py
"""
from typing import Optional
import ezpz
from tensorflow import device
import torch
from dataclasses import dataclass


def get_default_event(
        device_type: Optional[str] = None,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False
):
    """
    Get default event
    """
    device_type = device_type or ezpz.get_torch_device_type()
    if device_type == 'cuda' and torch.cuda.is_available():
        return torch.cuda.Event(
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess
        )
    elif device_type == 'xpu' and torch.xpu.is_available():
        return torch.xpu.Event(
            enable_timing=enable_timing,
        )
    elif device_type == 'mps':
        return torch.mps.Event(
            enable_timing=enable_timing,
        )
    else:
        return torch.cpu.Event()


@dataclass
class Event:
    """
    Event class
    """
    enable_timing: bool = False
    blocking: bool = False
    interprocess: bool = False

    def __post_init__(self):
        self.device_type = ezpz.get_torch_device_type()
        self.rank = ezpz.get_rank()
        self.local_rank = ezpz.get_local_rank()
        self.device = torch.device(f'{self.device_type}:{self.local_rank}')
        self.event = get_default_event()

    def __new__(cls, enable_timing: bool = False):
        return event.__new__(cls, enable_timing=enable_timing)
