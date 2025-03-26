# API Reference

This page provides a comprehensive API reference for the ezpz library, organized by module. It includes descriptions, parameters, return values, and usage examples for all public classes, functions, and methods.

## Table of Contents

- [ezpz](#ezpz)
- [ezpz.dist](#ezpzdist)
- [ezpz.configs](#ezpzconfigs)
- [ezpz.utils](#ezpzutils)

## ezpz

The main module of the ezpz library.

### Functions

#### `seed_everything(seed: int)`

Set random seeds for reproducibility.

Parameters:
- `seed` (int): The seed value to use for random number generation.

Example:
```python
import ezpz

ezpz.seed_everything(42)
```

#### `get_dist_info(framework: Optional[str] = None, verbose: Optional[bool] = None, hostfile: Optional[PathLike] = None) -> dict[str, str | int | list]`

Get distributed training information.

Parameters:
- `framework` (Optional[str]): The deep learning framework being used.
- `verbose` (Optional[bool]): Whether to print verbose output.
- `hostfile` (Optional[PathLike]): Path to the hostfile.

Returns:
- dict: A dictionary containing distributed training information.

Example:
```python
import ezpz

dist_info = ezpz.get_dist_info(framework='pytorch', verbose=True)
print(dist_info)
```

#### `print_dist_setup(framework: Optional[str] = None, hostfile: Optional[PathLike] = None) -> str`

Print the distributed setup information.

Parameters:
- `framework` (Optional[str]): The deep learning framework being used.
- `hostfile` (Optional[PathLike]): Path to the hostfile.

Returns:
- str: A string representation of the distributed setup.

Example:
```python
import ezpz

setup_info = ezpz.print_dist_setup(framework='pytorch')
print(setup_info)
```

#### `setup(framework: str = 'pytorch', backend: str = 'DDP', port: str = '5432', seed: Optional[int] = None, precision: Optional[str] = None, ngpus: Optional[int] = None)`

Set up the distributed training environment.

Parameters:
- `framework` (str): The deep learning framework to use (default: 'pytorch').
- `backend` (str): The distributed backend to use (default: 'DDP').
- `port` (str): The port to use for communication (default: '5432').
- `seed` (Optional[int]): Random seed for reproducibility.
- `precision` (Optional[str]): Precision to use for training.
- `ngpus` (Optional[int]): Number of GPUs to use.

Example:
```python
import ezpz

ezpz.setup(framework='pytorch', backend='deepspeed', seed=42)
```

#### `cleanup()`

Clean up the distributed training environment.

Example:
```python
import ezpz

# After training is complete
ezpz.cleanup()
```

## ezpz.dist

Module for distributed training utilities.

### Functions

#### `get_rank() -> int`

Get the current process rank.

Returns:
- int: The rank of the current process.

Example:
```python
from ezpz import dist

rank = dist.get_rank()
print(f"Current rank: {rank}")
```

#### `get_world_size(total: Optional[bool] = None, in_use: Optional[bool] = None) -> int`

Get the world size (total number of processes).

Parameters:
- `total` (Optional[bool]): If True, return the total available world size.
- `in_use` (Optional[bool]): If True, return the world size currently in use.

Returns:
- int: The world size.

Example:
```python
from ezpz import dist

world_size = dist.get_world_size()
print(f"World size: {world_size}")
```

#### `get_local_rank() -> int`

Get the local rank of the current process.

Returns:
- int: The local rank of the current process.

Example:
```python
from ezpz import dist

local_rank = dist.get_local_rank()
print(f"Local rank: {local_rank}")
```

#### `init_process_group(rank: int | str, world_size: int | str, timeout: str | int | timedelta) -> None`

Initialize the process group for distributed training.

Parameters:
- `rank` (int | str): The rank of the current process.
- `world_size` (int | str): The total number of processes.
- `timeout` (str | int | timedelta): Timeout for the initialization process.

Example:
```python
from ezpz import dist
from datetime import timedelta

dist.init_process_group(rank=0, world_size=4, timeout=timedelta(minutes=30))
```

## ezpz.configs

Module for configuration management.

### Classes

#### `TrainConfig`

A dataclass for storing training configuration.

Attributes:
- `gas` (int): Gradient accumulation steps.
- `framework` (str): The deep learning framework to use.
- `backend` (str): The distributed backend to use.
- `use_wandb` (bool): Whether to use Weights & Biases for logging.
- `seed` (Optional[int]): Random seed for reproducibility.
- `port` (Optional[str]): Port for communication.
- `dtype` (Optional[Any]): Data type for training.
- `load_from` (Optional[str]): Path to load a model from.
- `save_to` (Optional[str]): Path to save the model to.
- `ds_config_path` (Optional[str]): Path to the DeepSpeed configuration file.
- `wandb_project_name` (Optional[str]): Name of the Weights & Biases project.
- `ngpus` (Optional[int]): Number of GPUs to use.

Example:
```python
from ezpz.configs import TrainConfig

config = TrainConfig(
    framework='pytorch',
    backend='deepspeed',
    use_wandb=True,
    seed=42,
    ngpus=4
)
print(config)
```

### Functions

#### `load_ds_config(fpath: Optional[Union[str, os.PathLike, Path]] = None) -> dict[str, Any]`

Load a DeepSpeed configuration from a file.

Parameters:
- `fpath` (Optional[Union[str, os.PathLike, Path]]): Path to the DeepSpeed configuration file.

Returns:
- dict[str, Any]: The loaded DeepSpeed configuration.

Example:
```python
from ezpz.configs import load_ds_config

ds_config = load_ds_config('path/to/ds_config.json')
print(ds_config)
```

#### `get_logging_config() -> dict`

Get the logging configuration.

Returns:
- dict: The logging configuration.

Example:
```python
from ezpz.configs import get_logging_config

log_config = get_logging_config()
print(log_config)
```

## ezpz.utils

Module for utility functions.

### Classes

#### `DistributedPdb`

A distributed-aware Python debugger.

Example:
```python
from ezpz.utils import DistributedPdb

# Set a breakpoint in distributed code
DistributedPdb().set_trace()
```

### Functions

#### `breakpoint(rank: int = 0)`

Set a breakpoint on a specific rank in distributed training.

Parameters:
- `rank` (int): The rank to set the breakpoint on (default: 0).

Example:
```python
from ezpz.utils import breakpoint

# Set a breakpoint on rank 0
breakpoint(rank=0)
```

#### `get_max_memory_allocated(device: torch.device) -> float`

Get the maximum memory allocated on a device.

Parameters:
- `device` (torch.device): The device to query.

Returns:
- float: The maximum memory allocated in bytes.

Example:
```python
import torch
from ezpz.utils import get_max_memory_allocated

device = torch.device('cuda:0')
max_mem = get_max_memory_allocated(device)
print(f"Maximum memory allocated: {max_mem / 1e9:.2f} GB")
```

#### `save_dataset(dataset: xr.Dataset, outdir: PathLike, use_hdf5: Optional[bool] = True, fname: Optional[str] = None, **kwargs) -> Path`

Save a dataset to a file.

Parameters:
- `dataset` (xr.Dataset): The dataset to save.
- `outdir` (PathLike): The output directory.
- `use_hdf5` (Optional[bool]): Whether to use HDF5 format (default: True).
- `fname` (Optional[str]): The filename to use.
- `**kwargs`: Additional keyword arguments to pass to the saving function.

Returns:
- Path: The path to the saved file.

Example:
```python
import xarray as xr
from ezpz.utils import save_dataset

dataset = xr.Dataset(...)
outfile = save_dataset(dataset, outdir='path/to/output', use_hdf5=True)
print(f"Dataset saved to: {outfile}")
```

This API reference provides an overview of the main components and functions available in the ezpz library. For more detailed information on specific modules or functions, please refer to the inline documentation in the source code.