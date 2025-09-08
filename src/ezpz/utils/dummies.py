"""
ezpz/utils/dummies.py
"""

from typing import Any

from dataclasses import dataclass


class DummyMPI:
    # COMM_WORLD = None
    class COMM_WORLD:
        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def Get_size():
            return 1


@dataclass
class DummyTorch:
    __version__ = "0.0.0"
    float32 = Any
    float64 = Any
    int32 = Any
    int64 = Any
    uint8 = Any
    bool = Any

    # and torch.get_default_dtype() != torch.float64
    @staticmethod
    def get_default_dtype():
        return None

    class distributed:
        @staticmethod
        def init_process_group(*args, **kwargs):
            pass

        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def get_rank():
            return 0

        @staticmethod
        def get_world_size():
            return 1

    class cpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def synchronize(device):
            pass

    class mps:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def synchronize():
            pass

    class xpu:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def synchronize(device):
            pass

    class cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def synchronize(device):
            pass

    class backends:
        class mps:
            @staticmethod
            def is_available():
                return False

        class cpu:
            @staticmethod
            def synchronize(device):
                pass

        class xpu:
            @staticmethod
            def synchronize(device):
                pass

        class cuda:
            @staticmethod
            def synchronize(device):
                pass

    class device:
        def __init__(self, device_type):
            self.device_type = device_type
