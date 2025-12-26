# Quickstart

🍋 `ezpz` provides a set of dynamic, light weight utilities that simplify
running experiments with distributed PyTorch.

These can be broken down, roughly into two distinct categories:

1. [**Shell Environment**](#environment-setup):
   Bash script at
   [ezpz/bin/`utils.sh`](https://github.com/saforem2/ezpz/blob/main/utils/utils.sh),

   Use via:

   ```Bash
   source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
   ```

   This script contains utilities for automatically:
   - Detecting(when running behind a job scheduler, e.g. {Slurm,
     PBS})
   - Automatically load appropriate modules
   - Automatic virtual environment setup

   ... and more!  
   Check out [🏖️ Shell Environment](./shell-environment.md) for additional
   information.

1. [Python Library](#python-library):

   ```python
   import ezpz

   rank = ezpz.setup_torch()
   model = t
   ```

1. [Environment Setup](#environment-setup) (optional if you have `torch` + `mpi4py`)
1. [Launching](#launching) and running distributed PyTorch code (_from python!_)
1. [Device Management](#device-management), and running on different
   {`cuda`, `xpu`, `mps`, `cpu`} devices
1. [Experiment Tracking](#experiment-tracking) and tools for automatically
   recording, saving and plotting metrics.

Each of these components are designed so that you can pick and choose only
those tools that are useful for you.

However, if you find yourself:

- Building, testing, training, evaluating, running, debugging,

So, if you're someone who is new to distributed PyTorch, or someone who isi

Components from any of these categories can be used individually and

These tools are _not_ a replacement for the

## Environment Setup

See [docs/shell-environment](./shell-environment.md)

## Launching

## Device Management
