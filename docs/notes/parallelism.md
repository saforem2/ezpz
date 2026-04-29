# Parallelism Support

ezpz supports four dimensions of parallelism for distributed training:

| Abbreviation | Full Name          | What is sharded                            |
|:------------:|--------------------|--------------------------------------------|
| **TP**       | Tensor Parallel    | Individual weight matrices (columns / rows) |
| **CP**       | Context Parallel   | Sequence / context dimension               |
| **PP**       | Pipeline Parallel  | Model layers across pipeline stages        |
| **DP**       | Data Parallel      | Training data (each replica sees different batches) |

These compose multiplicatively — the total world size is always:

$$
W = \text{TP} \times \text{CP} \times \text{PP} \times \text{DP}
$$

DP is derived automatically:
$\text{DP} = W / (\text{TP} \times \text{CP} \times \text{PP})$.

## Quick Start

```bash
# 24 GPUs, TP=2, CP=2, PP=1 → DP=6
ezpz test --tp 2 --cp 2

# 24 GPUs, TP=1, CP=4, PP=2 → DP=3
ezpz test --tp 1 --cp 4 --pp 2
```

Under the hood, `ezpz test` calls
[`setup_torch()`][ezpz.distributed.setup_torch] with
`tensor_parallel_size`, `pipeline_parallel_size`, and
`context_parallel_size`.  When any of these is greater than 1, the call
chains into [`ezpz.tp.initialize_tensor_parallel()`][ezpz.tp.initialize_tensor_parallel]
to create the four process groups.

## How Process Groups Are Created

`initialize_tensor_parallel()` (in `src/ezpz/tp/__init__.py`) reshapes the
flat list of ranks into a 4-D tensor:

```python
groups = torch.arange(world_size).reshape(
    dp_size,                   # axis 0
    pipeline_parallel_size,    # axis 1
    context_parallel_size,     # axis 2
    tensor_parallel_size,      # axis 3
)
```

Each parallelism group is formed by slicing along the corresponding axis
while holding the other three fixed:

| Group | Slice              | Varies   | Fixed        |
|-------|--------------------|----------|--------------|
| DP    | `groups[:, i, j, k]` | DP axis  | PP, CP, TP |
| PP    | `groups[i, :, j, k]` | PP axis  | DP, CP, TP |
| CP    | `groups[i, j, :, k]` | CP axis  | DP, PP, TP |
| TP    | `groups[i, j, k, :]` | TP axis  | DP, PP, CP |

## The `ezpz.tp` Module

The `ezpz.tp` package provides Megatron-style tensor-parallel primitives
(adapted from [fairscale](https://github.com/facebookresearch/fairscale)):

| Sub-module | Contents |
|------------|----------|
| [`ezpz.tp`](../python/Code-Reference/tp/index.md) | `initialize_tensor_parallel()`, group accessors (`get_tensor_parallel_group()`, etc.), `destroy_tensor_parallel()` |
| [`ezpz.tp.layers`](../python/Code-Reference/tp/layers.md) | `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`, `ParallelEmbedding` |
| [`ezpz.tp.mappings`](../python/Code-Reference/tp/mappings.md) | Autograd communication functions: `copy_to_tensor_parallel_region`, `reduce_from_tensor_parallel_region`, `scatter_to_tensor_parallel_region`, `gather_from_tensor_parallel_region` |
| [`ezpz.tp.utils`](../python/Code-Reference/tp/utils.md) | `ensure_divisibility`, `split_tensor_along_last_dim`, `VocabUtility` |

## Model Wrapping

[`wrap_model()`][ezpz.distributed.wrap_model] dispatches based on the
`use_fsdp` flag:

| Path       | Wrapper | When |
|------------|---------|------|
| FSDP       | `torch.distributed.fsdp.FullyShardedDataParallel` | Default (`use_fsdp=True`) |
| DDP        | `torch.nn.parallel.DistributedDataParallel` | `use_fsdp=False` |
| DeepSpeed  | `deepspeed.initialize()` | Used directly in user code (bypasses `wrap_model`) |
| FSDP + TP  | `parallelize_module()` + FSDP on the DP mesh dimension | See the [`fsdp_tp` example](../examples/fsdp-tp.md) |

If `world_size <= 1`, `wrap_model()` returns the model unwrapped.

## CLI Flags

Defined in `src/ezpz/cli/flags.py`:

| Flag   | Default | Description              |
|--------|---------|--------------------------|
| `--tp` | 1       | Tensor parallel size     |
| `--pp` | 1       | Pipeline parallel stages |
| `--cp` | 1       | Context parallel size    |

## Partition Table

For a 24-GPU allocation, the valid TP / CP / PP / DP combinations are:

| World Size | TP | CP | PP | DP |
|:----------:|:--:|:--:|:--:|:--:|
|     24     |  1 |  1 |  1 | 24 |
|     24     |  2 |  1 |  1 | 12 |
|     24     |  1 |  2 |  1 | 12 |
|     24     |  1 |  1 |  2 | 12 |
|     24     |  2 |  2 |  1 |  6 |
|     24     |  2 |  1 |  2 |  6 |
|     24     |  1 |  2 |  2 |  6 |
|     24     |  4 |  1 |  1 |  6 |
|     24     |  1 |  4 |  1 |  6 |
|     24     |  1 |  1 |  4 |  6 |
|     24     |  4 |  2 |  1 |  3 |
|     24     |  4 |  1 |  2 |  3 |
|     24     |  2 |  4 |  1 |  3 |
|     24     |  2 |  1 |  4 |  3 |
|     24     |  1 |  4 |  2 |  3 |
|     24     |  1 |  2 |  4 |  3 |
|     24     |  4 |  2 |  2 |  3 |
|     24     |  2 |  4 |  2 |  3 |
|     24     |  2 |  2 |  2 |  3 |

## Examples from Aurora

??? example "TP=1, CP=4, PP=2, DP=3"

    ```bash
    $ launch python3 -Wignore -m ezpz.examples.test --tp 1 --cp 4 --pp 2
    [2024-12-31 15:36:13.333215][INFO][__init__.py:146] - > initializing model parallel with size 1
    [2024-12-31 15:36:13.333942][INFO][__init__.py:151] - > initializing context parallel with size 4
    [2024-12-31 15:36:13.334476][INFO][__init__.py:156] - > initializing pipeline with size 2
    [2024-12-31 15:36:13.334971][INFO][__init__.py:159] - > initializing ddp with size 3
    [2024-12-31 15:36:14.402809][INFO][dist.py:846] - [ 0/23]: [cp:0/3][pp:0/1][dp:0/2]
    [2024-12-31 15:36:14.402209][INFO][dist.py:846] - [ 3/23]: [cp:3/3][pp:0/1][dp:0/2]
    [2024-12-31 15:36:14.402211][INFO][dist.py:846] - [ 1/23]: [cp:1/3][pp:0/1][dp:0/2]
    [2024-12-31 15:36:14.402197][INFO][dist.py:846] - [ 7/23]: [cp:3/3][pp:1/1][dp:0/2]
    [2024-12-31 15:36:14.402239][INFO][dist.py:846] - [ 4/23]: [cp:0/3][pp:1/1][dp:0/2]
    ...
    ```

??? example "TP=CP=PP=2, DP=3"

    ```bash
    $ launch python3 -Wignore -m ezpz.examples.test --tp 2 --cp 2 --pp 2
    [2024-12-31 15:19:37.033562][INFO][__init__.py:146] - > initializing model parallel with size 2
    [2024-12-31 15:19:37.034083][INFO][__init__.py:151] - > initializing context parallel with size 2
    [2024-12-31 15:19:37.034451][INFO][__init__.py:156] - > initializing pipeline with size 2
    [2024-12-31 15:19:37.034792][INFO][__init__.py:159] - > initializing ddp with size 3
    [2024-12-31 15:19:38.239822][INFO][dist.py:824] - Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
    [2024-12-31 15:19:38.240412][INFO][dist.py:846] - [ 0/23]: [cp:0/1][pp:0/1][tp:0/1][dp:0/2]
    ...
    ```

??? example "TP=CP=2, PP=1, DP=6"

    ```bash
    $ launch python3 -Wignore -m ezpz.examples.test --tp 2 --cp 2
    [2024-12-31 15:29:21.697491][INFO][__init__.py:146] - > initializing model parallel with size 2
    [2024-12-31 15:29:21.698012][INFO][__init__.py:151] - > initializing context parallel with size 2
    [2024-12-31 15:29:21.698377][INFO][__init__.py:156] - > initializing pipeline with size 1
    [2024-12-31 15:29:21.698745][INFO][__init__.py:159] - > initializing ddp with size 6
    [2024-12-31 15:29:22.900343][INFO][dist.py:846] - [ 0/23]: [cp:0/1][tp:0/1][dp:0/5]
    ...
    ```
