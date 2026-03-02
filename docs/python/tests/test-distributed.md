# Test Distributed

Tests for the [`ezpz.distributed`](../Code-Reference/distributed.md) module.

- See [`tests/test_distributed.py`](https://github.com/saforem2/ezpz/blob/main/tests/test_distributed.py)

## Overview

The `test_distributed.py` suite contains 148 tests across 29 test classes,
covering every public function in `ezpz.distributed` plus critical private
helpers. All tests run without a real distributed environment by using mocks
and the `mock_dist_env` / `mock_pbs_env` fixtures from `conftest.py`.

## Running

```bash
# Run only distributed tests
python -m pytest tests/test_distributed.py -v

# Run a specific test class
python -m pytest tests/test_distributed.py::TestSetupTorch -v
```

## Test Classes

| Class | Covers |
|-------|--------|
| `TestGetRank` | `get_rank` — MPI, env-var, and fallback paths |
| `TestGetWorldSize` | `get_world_size` — MPI, env-var, and fallback paths |
| `TestGetLocalRank` | `get_local_rank` — MPI, env-var, and fallback paths |
| `TestGetTorchDeviceType` | `get_torch_device_type` — CUDA, XPU, MPS, CPU detection |
| `TestGetTorchDevice` | `get_torch_device` — device string construction |
| `TestGetTorchBackend` | `get_torch_backend` — NCCL, CCL, Gloo selection |
| `TestSetupTorch` | `setup_torch` — full initialization flow |
| `TestSynchronize` | `synchronize` — device synchronization |
| `TestBarrier` | `barrier` — distributed barrier |
| `TestCleanup` | `cleanup` — `destroy_process_group` call |
| `TestTimeitlogit` | `timeitlogit` — context-manager timing decorator |
| `TestSetupWandb` | `setup_wandb` — W&B initialization |
| `TestVerifyWandb` | `verify_wandb` — W&B run verification |
| `TestGetDistInfo` | `get_dist_info` — `DistInfo` namedtuple |
| `TestBroadcast` | `broadcast` — dictionary broadcast |
| `TestAllReduce` | `all_reduce` — tensor all-reduce |
| `TestWrapModel` | `wrap_model` — DDP / FSDP wrapping dispatch |
| `TestWrapModelForDdp` | `wrap_model_for_ddp` — DDP-specific wrapping |
| `TestGetMachine` | `get_machine` — machine name detection |
| `TestGetGpusPerNode` | `get_gpus_per_node` — GPU count per node |
| `TestGetHostname` | `get_hostname` — hostname retrieval |
| `TestGetNumNodes` | `get_num_nodes` — node count |
| `TestGetNodeIndex` | `get_node_index` — node index |
| `TestSeedEverything` | `seed_everything` — RNG seeding |
| `TestQueryEnvironment` | `query_environment` — env-var dict |
| `TestPrintDistSetup` | `print_dist_setup` — formatted output |
| `TestLogDictAsBulletedList` | `log_dict_as_bulleted_list` — logging helper |
| `TestExpandSlurmNodelist` | `_expand_slurm_nodelist` — SLURM nodelist parsing |
| `TestHostfileHelpers` | hostfile read / write / fallback helpers |
