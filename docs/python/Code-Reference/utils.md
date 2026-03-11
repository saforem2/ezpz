# `ezpz.utils`

- See [ezpz/`utils`](https://github.com/saforem2/ezpz/tree/main/src/ezpz/utils)

Utility functions for debugging, GPU memory monitoring, DeepSpeed configuration,
and dataset I/O.

## Distributed Debugging

`ezpz` provides debugger classes that work correctly in multi-process
distributed environments.

### `DistributedPdb`

A `pdb.Pdb` subclass that synchronizes all ranks at breakpoints, allowing
interactive debugging of a single rank while others wait at a barrier.

### `ForkedPdb`

A `pdb.Pdb` subclass that works in forked (multiprocessing) contexts by
redirecting stdin.

### `breakpoint(rank=0)`

Set a breakpoint that only activates on the specified rank. All other ranks
wait at a barrier until the debugger continues.

??? example "Debugging rank 0 in a distributed job"

    ```python
    import ezpz
    from ezpz.utils import breakpoint

    rank = ezpz.setup_torch()

    # Only rank 0 drops into the debugger; others wait
    breakpoint(rank=0)

    # All ranks continue together after the debugger exits
    output = model(input_data)
    ```

## GPU Memory Monitoring

```python
from ezpz.utils import get_max_memory_allocated, get_max_memory_reserved

device = ezpz.get_torch_device()
peak_allocated = get_max_memory_allocated(device)  # in bytes
peak_reserved = get_max_memory_reserved(device)    # in bytes

print(f"Peak allocated: {peak_allocated / 1e9:.2f} GB")
print(f"Peak reserved:  {peak_reserved / 1e9:.2f} GB")
```

Both functions work with CUDA and XPU devices, falling back gracefully on
unsupported platforms.

## Peak FLOPS Lookup

Look up the theoretical peak BF16 FLOPS for known GPU types:

```python
from ezpz.utils import get_peak_flops

flops = get_peak_flops("A100")           # NVIDIA A100
flops = get_peak_flops("H100 SXM")       # NVIDIA H100 SXM
flops = get_peak_flops("H100 NVL")       # NVIDIA H100 NVL variant
flops = get_peak_flops("H200")           # NVIDIA H200
flops = get_peak_flops("B200")           # NVIDIA B200
flops = get_peak_flops("Max 1550")       # Intel Data Center GPU Max (PVC)
```

Note: The lookup is **case-sensitive** and uses substring matching (e.g.
`"A100" in device_name`). On systems with `lspci` available, the function
will auto-detect the GPU name from PCI device listings.

## DeepSpeed Config Generators

Generate DeepSpeed configuration dictionaries for various ZeRO stages and
precision modes:

??? example "ZeRO Stage 1/2 auto config"

    ```python
    from ezpz.utils import write_deepspeed_zero12_auto_config

    # Returns config dict and writes JSON to output_dir
    config = write_deepspeed_zero12_auto_config(
        zero_stage=2,
        output_dir="./ds_configs"
    )
    ```

??? example "ZeRO Stage 3 auto config"

    ```python
    from ezpz.utils import write_deepspeed_zero3_auto_config

    config = write_deepspeed_zero3_auto_config(
        zero_stage=3,
        output_dir="./ds_configs"
    )
    ```

??? example "Precision configs"

    ```python
    from ezpz.utils import get_bf16_config_json, get_fp16_config_json

    bf16_config = get_bf16_config_json(enabled=True)
    # {"enabled": True}

    fp16_config = get_fp16_config_json(enabled=True)
    # {"enabled": True}
    ```

??? example "Full DeepSpeed config"

    ```python
    from ezpz.utils import get_deepspeed_config_json

    config = get_deepspeed_config_json(
        auto_config=True,
        gradient_accumulation_steps=4,
        stage=2,
        output_dir="./ds_configs",
    )
    ```

??? example "FLOPs profiler config"

    ```python
    from ezpz.utils import get_flops_profiler_config_json

    profiler_config = get_flops_profiler_config_json(
        enabled=True,
        profile_step=1,
        module_depth=-1,
        top_modules=1,
        detailed=True,
    )
    ```

## Dataset I/O

Save and load `xarray.Dataset` objects to/from HDF5 files:

```python
from ezpz.utils import save_dataset, dataset_to_h5pyfile, dataset_from_h5pyfile

# Save a dataset (uses HDF5 by default)
path = save_dataset(dataset, outdir="./data", fname="metrics.h5")

# Direct HDF5 operations
dataset_to_h5pyfile("metrics.h5", dataset)
loaded = dataset_from_h5pyfile("metrics.h5")
```

::: ezpz.utils
