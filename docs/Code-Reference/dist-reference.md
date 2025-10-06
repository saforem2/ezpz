# `ezpz.dist`

Distributed computing utilities for the ezpz package.

- See: [`ezpz.dist`](https://github.com/saforem2/ezpz/blob/main/ezpz/dist.py)

## Overview

The `ezpz.dist` module provides utilities for distributed computing, including
process management, device detection, and distributed training setup.

## Key Functions

::: ezpz.dist

## Usage Examples

### Process Information

```python
import ezpz.dist as dist

# Get process rank and world size
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = dist.get_local_rank()

print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
```

### Device Management

```python
import ezpz.dist as dist

# Get device information
device_type = dist.get_torch_device_type()
device = dist.get_torch_device()

print(f"Device Type: {device_type}")  # "cpu", "cuda", "xpu", "mps"
print(f"Device: {device}")            # torch.device object
```

### Distributed Training Setup

```python
import ezpz.dist as dist

# Setup distributed training
setup_info = dist.setup_torch(
    backend="ddp",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    context_parallel_size=1
)

print(f"Setup Info: {setup_info}")
```

### Environment Information

```python
import ezpz.dist as dist

# Get distributed environment information
env_info = dist.query_environment()
dist_info = dist.get_dist_info()

print(f"Environment Info: {env_info}")
print(f"Distributed Info: {dist_info}")
```

### Random Seed Management

```python
import ezpz.dist as dist

# Set random seed for reproducibility
dist.seed_everything(42)
```
