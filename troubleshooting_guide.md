# Troubleshooting Guide

This guide provides solutions for common issues you might encounter when using ezpz, including distributed training problems, configuration errors, and performance issues.

## Table of Contents
1. [Distributed Training Issues](#distributed-training-issues)
2. [Configuration Errors](#configuration-errors)
3. [Performance Issues](#performance-issues)

## Distributed Training Issues

### Error: Unable to initialize the process group

If you encounter an error like this:

```
RuntimeError: Unable to initialize process group: NCCL error: unhandled system error
```

**Solution:**
1. Check if all nodes can communicate with each other.
2. Ensure that the NCCL backend is properly installed and configured.
3. Try setting the `NCCL_DEBUG` environment variable to get more information:

   ```bash
   export NCCL_DEBUG=INFO
   ```

4. If you're not using an Infiniband interconnect, disable it:

   ```bash
   export NCCL_IB_DISABLE=1
   ```

### Error: Address already in use

If you see an error message like:

```
RuntimeError: Address already in use
```

**Solution:**
1. Change the port number in your configuration or when initializing the distributed setup.
2. In the `setup_torch` function, modify the `port` parameter:

   ```python
   setup_torch(backend='DDP', port='5678')
   ```

## Configuration Errors

### Error: Unable to parse backend

If you encounter an error like:

```
ValueError: Unable to parse backend: invalid_backend
```

**Solution:**
1. Check the `backend` parameter in your configuration.
2. Ensure you're using one of the supported backends: 'DDP', 'deepspeed', or 'horovod'.
3. Example of correct usage:

   ```python
   setup_torch(backend='DDP')
   ```

### Error: Mismatch in total world size

If you see a warning or error about mismatched world sizes:

```
WARNING: Mismatch in `ngpus_in_use` and `ngpus_available` 8 vs. 4
```

**Solution:**
1. Check your configuration to ensure you've specified the correct number of GPUs.
2. Verify that all nodes have the same number of available GPUs.
3. Use the `get_world_size` function to check the total number of GPUs:

   ```python
   from ezpz import get_world_size
   print(f"Total GPUs: {get_world_size(total=True)}")
   print(f"GPUs in use: {get_world_size(in_use=True)}")
   ```

## Performance Issues

### Slow Training Speed

If you're experiencing slower than expected training speeds:

**Solution:**
1. Check your batch size and adjust it if necessary.
2. Verify that you're using the correct device (CPU/GPU/XPU).
3. Monitor GPU utilization using `nvidia-smi` or similar tools.
4. Use the `timeitlogit` decorator to profile your code:

   ```python
   from ezpz import timeitlogit

   @timeitlogit(rank=0, verbose=True)
   def my_function():
       # Your code here
       pass
   ```

5. Consider using mixed-precision training if you're not already:

   ```python
   from torch.cuda.amp import GradScaler

   scaler = GradScaler(enabled=(config.train.dtype == 'float16'))
   ```

### Out of Memory Errors

If you encounter out of memory errors:

**Solution:**
1. Reduce your batch size.
2. Use gradient accumulation to simulate larger batch sizes:

   ```python
   config.optimizer.gas = 4  # Gradient Accumulation Steps
   ```

3. Monitor memory usage using `get_max_memory_allocated`:

   ```python
   from ezpz.utils import get_max_memory_allocated
   import torch

   print(f"Max memory allocated: {get_max_memory_allocated(torch.device('cuda'))}")
   ```

4. If using DeepSpeed, adjust the ZeRO optimization levels in your DeepSpeed configuration.

Remember to check the ezpz documentation for more detailed information on configuration options and best practices. If you continue to experience issues, please file a bug report on the project's issue tracker with a detailed description of the problem and steps to reproduce it.