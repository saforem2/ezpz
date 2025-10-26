---
title: Getting Started with ezpz
description: A comprehensive guide to installing and using ezpz for distributed training
---

# Getting Started with ezpz

Welcome to ezpz, a Python library designed to simplify distributed training across various frameworks and backends. This guide will help you get started with ezpz, covering installation, basic usage, and an overview of its capabilities.

## Table of Contents

1. [Installation](#installation)
2. [Supported Frameworks and Backends](#supported-frameworks-and-backends)
3. [Basic Usage](#basic-usage)
4. [Distributed Training Setup](#distributed-training-setup)
5. [Advanced Features](#advanced-features)

## Installation

To install ezpz, use the following pip command:

```bash
python3 -m pip install -e "git+https://github.com/saforem2/ezpz#egg=ezpz" --require-virtualenv
```

It's recommended to install ezpz within a virtual environment for better package management.

## Supported Frameworks and Backends

ezpz supports the following frameworks and distributed training backends:

- PyTorch
  - DDP (DistributedDataParallel)
  - DeepSpeed
  - Horovod
- TensorFlow
  - Horovod

## Basic Usage

To use ezpz in your project, start by importing it:

```python
import ezpz as ez
```

### Setting up the Environment

ezpz provides utility functions to set up your distributed training environment:

```python
# For PyTorch
rank = ez.setup_torch(backend='DDP')  # or 'deepspeed' or 'horovod'

# For TensorFlow
rank = ez.setup_tensorflow()
```

### Getting Distributed Information

You can easily access information about your distributed setup:

```python
world_size = ez.get_world_size()
local_rank = ez.get_local_rank()
device = ez.get_torch_device()
```

## Distributed Training Setup

ezpz simplifies the process of setting up distributed training. Here's an example of how to use it with PyTorch:

```python
import torch
import ezpz as ez

def main():
    # Set up the distributed environment
    rank = ez.setup_torch(backend='DDP')
    
    # Your model setup
    model = YourModel().to(ez.get_torch_device())
    model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Your training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Training steps
            ...
    
    # Clean up
    ez.cleanup()

if __name__ == '__main__':
    main()
```

## Advanced Features

### Tensor Parallel Training

ezpz supports tensor parallel training. Here's how to set it up:

```python
rank = ez.setup_torch(
    backend='DDP',
    tensor_parallel_size=4,
    pipeline_parallel_size=2
)
```

### Wandb Integration

ezpz provides easy integration with Weights & Biases for experiment tracking:

```python
ez.setup_wandb(project_name='my_project')
```

### Shell Utilities

ezpz also includes shell utilities to help with job setup and environment configuration. You can source these utilities in your shell:

```bash
source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
```

Then, you can use functions like:

```bash
ezpz_setup_env
```

This will automatically set up your Python environment and configure your job for distributed training.

## Conclusion

This guide covers the basics of getting started with ezpz. For more detailed information on specific features or advanced usage, please refer to the other documentation pages or the [ezpz GitHub repository](https://github.com/saforem2/ezpz).

Remember, ezpz is designed to make distributed training easier across different frameworks and backends. Whether you're using PyTorch or TensorFlow, on CPUs or GPUs, ezpz provides a consistent interface to help you focus on your machine learning tasks rather than infrastructure setup.