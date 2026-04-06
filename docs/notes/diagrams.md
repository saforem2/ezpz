# Repository Overview

> **"Write once, run anywhere"** — Portable distributed PyTorch across any
> supported hardware (NVIDIA, AMD, Intel, MPS, CPU) with **zero code changes**.

`ezpz` is a library that abstracts away the complexity of distributed training
in PyTorch. It auto-detects hardware, selects the right communication backend,
discovers node topology via MPI, and wraps models for data/tensor parallelism —
all behind a single function call.

## High-Level Architecture

The codebase is organized around a central **distributed training core**
(`distributed.py`) that everything else plugs into. The CLI and job scheduler
modules feed into the core, which in turn drives model wrapping and tensor
parallelism. Logging, diagnostics, and configuration live on the periphery and
are consumed as needed.

```mermaid
flowchart TD
    cli["CLI / Launch - ezpz launch, test, doctor"]
    core["Distributed Training Core - distributed.py"]
    tp_block["Tensor Parallel - tp/"]
    sched["Job Schedulers - slurm.py, pbs.py, jobs.py"]
    wrap["Model Wrapping - DDP, FSDP, DeepSpeed, Horovod"]
    data_block["Data Loading - data/"]
    config["Configs and Utilities - configs.py, utils/, lazy.py"]
    logging_block["Logging and Monitoring - log/, history.py"]
    diag["Diagnostics - doctor.py, profile.py"]

    cli --> core
    sched --> core
    config --> core
    logging_block --> core
    diag --> core
    core --> wrap
    core --> tp_block
    data_block --> tp_block
```

## Core Initialization Flow

The central entry point is `setup_torch()` — a single function call that
bootstraps distributed training regardless of hardware, scheduler, or
framework:

```mermaid
flowchart TD
    user["rank = ezpz.setup_torch()"]
    mpi["MPI Discovery: get_rank / get_world_size / get_local_rank"]
    addr["Master Addr+Port: rank 0 broadcasts hostname and free port"]
    device["Device Detection: CUDA / XPU / MPS / CPU"]
    backend["Backend Selection: NCCL / XCCL / CCL / Gloo"]
    fw["Framework Init: DDP / DeepSpeed / Horovod"]
    tppp["Optional TP/PP/CP: create orthogonal process groups"]
    seed["Seed RNGs: seed x rank+1"]
    ret(["return rank"])

    user --> mpi --> addr --> device --> backend --> fw
    fw --> tppp --> seed --> ret
```

## Lazy Loading Architecture

`import ezpz` is near-instant. Heavy dependencies (`torch`, `mpi4py`) are
deferred until first use via `__getattr__` on the package. The first time you
access an attribute like `ezpz.setup_torch`, the package walks a search order
of submodules, imports the one that defines it, caches the result, and returns
it. Every subsequent access hits the cache with no import overhead.

```mermaid
sequenceDiagram
    participant User
    participant Init as ezpz.__init__
    participant Cache as _IMPORT_CACHE
    participant Mod as ezpz.distributed

    User->>Init: import ezpz (instant, no torch/mpi)
    Note over Init: Only loads __about__.__version__

    User->>Init: ezpz.setup_torch()
    Init->>Init: __getattr__("setup_torch")
    Init->>Init: Check _LAZY_MODULES -- not found
    Init->>Init: Walk _MODULE_SEARCH_ORDER

    loop For each module in search order
        Init->>Cache: Already loaded?
        alt Cache miss
            Init->>Mod: importlib.import_module()
            Note over Mod: torch, mpi4py loaded HERE
            Mod-->>Cache: Store in cache
        end
        Init->>Mod: hasattr(module, "setup_torch")?
    end

    Mod-->>Init: Return setup_torch function
    Init-->>User: Cached -- subsequent access instant
```

## Module Dependency Graph

The graph below shows which modules import from each other. `distributed.py`
is the most depended-on module — it provides rank, device, and backend
primitives that almost everything else needs. External dependencies on `torch`
and `mpi4py` are confined to `distributed.py` and `tp/`, keeping the rest of
the package lightweight.

```mermaid
flowchart TD
    init["__init__.py - lazy __getattr__"]

    init --> configs
    init --> distributed
    init --> tp
    init --> launch
    init --> jobs
    init --> log
    init --> history
    init --> doctor
    init --> profile
    init --> models
    init --> tplot
    init --> utils

    distributed --> configs
    distributed --> tp
    distributed --> log

    launch --> pbs
    launch --> slurm
    launch --> configs

    jobs --> pbs
    jobs --> distributed

    pbs --> distributed
    pbs --> slurm

    history --> distributed
    history --> tplot
    history --> utils

    doctor --> distributed

    data_mod["data/"] --> distributed
    data_mod --> tp

    tp --> torch_dist["torch.distributed"]
    distributed --> torch_dist
    distributed --> mpi["mpi4py"]

    log --> distributed

    style init fill:#4a9eff,color:#fff
    style distributed fill:#2ecc71,color:#fff
    style tp fill:#e67e22,color:#fff
    style launch fill:#9b59b6,color:#fff
```

## Hardware & Backend Support Matrix

| | NVIDIA (CUDA) | AMD (ROCm) | Intel (XPU) | Apple (MPS) | CPU |
|---|:---:|:---:|:---:|:---:|:---:|
| **NCCL** | ✅ | — | — | — | — |
| **XCCL** | — | — | ✅ | — | — |
| **CCL** | — | — | ✅* | — | — |
| **Gloo** | ✅ | ✅ | ✅ | ✅ | ✅ |

\* CCL is the fallback when XCCL is unavailable on Intel XPU.

## Device & Backend Detection

When `setup_torch()` is called, ezpz needs to answer two questions: *what
hardware am I running on?* and *which communication backend should I use?*

The `get_torch_device_type()` and `get_torch_backend()` functions handle this
by probing available hardware in a fixed priority order. Environment variables
(`TORCH_DEVICE`, `TORCH_BACKEND`) can override auto-detection when needed — for
example, forcing `cpu` on a GPU node for debugging.

```mermaid
flowchart LR
    subgraph Device["get_torch_device_type()"]
        direction TB
        env_dev{"TORCH_DEVICE - env var?"}
        xpu{"torch.xpu - available?"}
        cuda{"torch.cuda - available?"}
        mps{"torch.backends.mps - available?"}
        cpu["cpu"]

        env_dev -- yes --> done_dev(["return env value"])
        env_dev -- no --> xpu
        xpu -- yes --> done_xpu(["xpu"])
        xpu -- no --> cuda
        cuda -- yes --> done_cuda(["cuda"])
        cuda -- no --> mps
        mps -- yes --> done_mps(["mps"])
        mps -- no --> cpu
    end

    subgraph Backend["get_torch_backend()"]
        direction TB
        env_be{"TORCH_BACKEND - env var?"}
        is_cuda{"device is - cuda?"}
        is_xpu{"device is - xpu?"}
        has_xccl{"xccl - available?"}
        gloo["gloo"]

        env_be -- yes --> done_be(["return env value"])
        env_be -- no --> is_cuda
        is_cuda -- yes --> nccl(["nccl"])
        is_cuda -- no --> is_xpu
        is_xpu -- yes --> has_xccl
        has_xccl -- yes --> xccl(["xccl"])
        has_xccl -- no --> ccl(["ccl"])
        is_xpu -- no --> gloo
    end
```

## Model Wrapping Decision Tree

After `setup_torch()` initializes the process group, models need to be wrapped
for distributed training. The high-level `wrap_model()` function makes this
decision automatically: if only one GPU is in use, the model is returned
unwrapped. Otherwise, it chooses between FSDP (shards parameters across ranks
for memory efficiency) and DDP (replicates the full model on each rank).

For finer control, `wrap_model_for_ddp()`, `wrap_model_for_fsdp()`, and
`wrap_model_for_fsdp2()` are available as explicit alternatives.

```mermaid
flowchart TD
    entry["wrap_model(model, use_fsdp, dtype)"]

    ws{"world_size > 1?"}
    entry --> ws

    ws -- No --> ret_plain(["return model unwrapped"])

    ws -- Yes --> fsdp_q{"use_fsdp?"}

    fsdp_q -- Yes --> fsdp_wrap["FSDP + MixedPrecision bf16/fp32"]
    fsdp_q -- No --> ddp_wrap["DDP with device_ids"]

    style entry fill:#4a9eff,color:#fff
    style fsdp_wrap fill:#2ecc71,color:#fff
    style ddp_wrap fill:#3498db,color:#fff
    style ret_plain fill:#95a5a6,color:#fff
```

## Multi-Dimensional Parallelism

For large models that don't fit on a single GPU, data parallelism alone isn't
enough. `ezpz` supports composing multiple parallelism strategies
simultaneously:

- **Tensor Parallelism (TP)**: Splits individual layers across GPUs within a
  node. Each GPU holds a shard of every weight matrix.
- **Data Parallelism (DP)**: Replicates the model across groups of GPUs, each
  processing different data.
- **Pipeline Parallelism (PP)**: Assigns different layers to different GPUs in
  a chain.
- **Context Parallelism (CP)**: Splits long sequences across GPUs.

These are set up via `setup_torch(tensor_parallel_size=...,
pipeline_parallel_size=..., context_parallel_size=...)`, which calls
`initialize_tensor_parallel()` to create orthogonal process groups.

The diagram below shows a 16-GPU layout with TP size 2 — each pair of GPUs
shares a model shard, while DP groups span across pairs:

```mermaid
flowchart TD
    subgraph world["World Size = 16 GPUs -- 2 nodes x 8 GPUs/node"]
        subgraph node0["Node 0"]
            direction LR
            subgraph tp0["TP Group 0"]
                g0["GPU 0"]
                g1["GPU 1"]
            end
            subgraph tp1["TP Group 1"]
                g2["GPU 2"]
                g3["GPU 3"]
            end
            subgraph tp2["TP Group 2"]
                g4["GPU 4"]
                g5["GPU 5"]
            end
            subgraph tp3["TP Group 3"]
                g6["GPU 6"]
                g7["GPU 7"]
            end
        end
        subgraph node1["Node 1"]
            direction LR
            subgraph tp4["TP Group 4"]
                g8["GPU 8"]
                g9["GPU 9"]
            end
            subgraph tp5["TP Group 5"]
                g10["GPU 10"]
                g11["GPU 11"]
            end
            subgraph tp6["TP Group 6"]
                g12["GPU 12"]
                g13["GPU 13"]
            end
            subgraph tp7["TP Group 7"]
                g14["GPU 14"]
                g15["GPU 15"]
            end
        end
    end

    dpnote["DP Groups cross-node: - 0,2,4,6,8,10,12,14 = TP rank 0 - 1,3,5,7,9,11,13,15 = TP rank 1"]

    style tp0 fill:#e67e22,color:#fff
    style tp1 fill:#e67e22,color:#fff
    style tp2 fill:#e67e22,color:#fff
    style tp3 fill:#e67e22,color:#fff
    style tp4 fill:#e67e22,color:#fff
    style tp5 fill:#e67e22,color:#fff
    style tp6 fill:#e67e22,color:#fff
    style tp7 fill:#e67e22,color:#fff
    style node0 fill:#dbeafe,stroke:#3498db
    style node1 fill:#d1fae5,stroke:#2ecc71
```

### Process Group Creation

`initialize_tensor_parallel(tp_size, pp_size, cp_size)` creates a 4D rank
tensor and slices it into orthogonal process groups:

```mermaid
flowchart LR
    init["initialize_tensor_parallel"]

    calc["dp_size = world_size / tp x pp x cp"]
    tensor["Build 4D rank tensor"]

    subgraph groups["Process Groups Created"]
        direction TB
        tpg["TP Group - ranks sharing a model shard"]
        dpg["DP Group - ranks with same TP position"]
        ppg["PP Group - pipeline stage chain"]
        cpg["CP Group - context/sequence splits"]
    end

    init --> calc --> tensor --> groups
```

## Tensor Parallel Layers

The `tp/` module provides drop-in replacements for standard PyTorch layers that
automatically split computation across TP ranks. Each layer pairs with autograd
functions from `tp/mappings.py` that insert the right collective operations
(all-reduce, scatter, gather) in the forward and backward passes so gradients
flow correctly across shards.

```mermaid
flowchart LR
    subgraph layers["tp/layers.py"]
        vpe["VocabParallelEmbedding - split along vocab dim"]
        pe["ParallelEmbedding - split along embed dim"]
        cpl["ColumnParallelLinear - Y = X * A, A split by columns"]
        rpl["RowParallelLinear - Y = X * A, A split by rows"]
    end

    subgraph mappings["tp/mappings.py autograd"]
        copy["copy_to_tp_region - fwd: identity - bwd: all-reduce"]
        reduce["reduce_from_tp_region - fwd: all-reduce - bwd: identity"]
        scatter["scatter_to_tp_region - fwd: split - bwd: gather"]
        gather["gather_from_tp_region - fwd: gather - bwd: split"]
    end

    cpl --> copy
    cpl --> gather
    rpl --> scatter
    rpl --> reduce
    vpe --> reduce
    pe --> copy
    pe --> gather
```

## Job Scheduler Integration

`ezpz launch` is the main entry point for running distributed jobs. It
auto-detects the active scheduler by checking for `PBS_JOBID` or `SLURM_JOB_ID`
environment variables, then builds the appropriate launch command (`mpiexec`,
`srun`, or local `mpirun`). On HPC systems, it reads the node allocation from
the scheduler, constructs a hostfile, and handles machine-specific quirks like
CPU binding and VNI settings.

If no scheduler is detected, it falls back to a local launch with `mpirun`,
auto-detecting the number of available GPUs.

```mermaid
flowchart TD
    cmd["ezpz launch -- python train.py"]

    detect{"Detect Scheduler"}
    cmd --> detect

    detect -- PBS_JOBID --> pbs
    detect -- SLURM_JOB_ID --> slurm
    detect -- Neither --> local

    subgraph pbs["PBS: Polaris, Aurora, Sophia"]
        pbs_env["PBS_NODEFILE, PBS_JOBID"]
        pbs_cmd["mpiexec / mpirun with CPU affinity"]
    end

    subgraph slurm["SLURM: Perlmutter, Frontier"]
        slurm_env["SLURM_NODELIST, SLURM_JOB_ID"]
        slurm_cmd["srun -u --verbose -N nodes -n gpus"]
    end

    subgraph local["Local / Laptop"]
        local_cmd["mpirun -np N auto-detect GPUs"]
    end

    pbs_env --> hostfile["Build hostfile, discover nodes, resolve master"]
    slurm_env --> hostfile
    pbs_cmd --> exec["Execute across all nodes"]
    slurm_cmd --> exec
    local_cmd --> exec

    persist["Job metadata persisted to ~/SCHEDULER-jobs/jobid/"]
    exec --> persist

    style cmd fill:#4a9eff,color:#fff
```

### Machine-Specific Launch Commands

Different HPC systems need different MPI flags. `pbs.build_launch_cmd()`
encodes these per-machine defaults so users don't have to remember them —
Aurora needs `--no-vni` and `--envall`, Polaris needs `--cpu-bind=depth`, and
Sophia uses `mpirun` instead of `mpiexec`.

```mermaid
flowchart LR
    build["pbs.build_launch_cmd()"]

    build --> sophia{"Sophia?"}
    build --> aurora{"Aurora / SunSpot?"}
    build --> polaris{"Polaris / Other?"}

    sophia -- yes --> mpirun["mpirun -n N -N PPn --hostfile=..."]
    aurora -- yes --> mpiexec_aurora["mpiexec --envall --np N --ppn PPn --no-vni"]
    polaris -- yes --> mpiexec_pol["mpiexec --cpu-bind=depth --depth=8"]
```

## Supported HPC Systems

`ezpz` has been tested on the following systems. Each has its own combination of
scheduler, accelerator, and communication backend — `ezpz` handles the
differences automatically.

```mermaid
mindmap
  root((ezpz))
    ALCF
      Aurora
        PBS
        Intel XPU
        XCCL
      Polaris
        PBS
        NVIDIA A100
        NCCL
      Sophia
        PBS
        NVIDIA
        NCCL
      Sirius
        PBS
        NVIDIA
        NCCL
      SunSpot
        PBS
        Intel XPU
        XCCL
    OLCF
      Frontier
        SLURM
        AMD MI250X
        Gloo
    NERSC
      Perlmutter
        SLURM
        NVIDIA A100
        NCCL
    Local
      Laptop or Workstation
        No scheduler
        MPS or CPU
        Gloo
```

## Logging Architecture

In distributed training, having every rank print to stdout creates unreadable
noise. `ezpz` provides rank-aware logging that suppresses output from non-rank-0
processes by default. Only rank 0 logs at the configured level; all other ranks
are set to `CRITICAL` (effectively silent). This can be overridden by setting
`LOG_FROM_ALL_RANKS=1` for debugging.

```mermaid
flowchart TD
    entry["ezpz.log.get_logger name"]
    check_rank{"get_rank"}
    entry --> check_rank
    check_rank -->|rank 0| r0["EZPZ_LOG_LEVEL - default: INFO"]
    check_rank -->|other ranks| rn{"LOG_FROM_ALL_RANKS - env var set?"}
    rn -->|Yes| log_all["EZPZ_LOG_LEVEL"]
    rn -->|No| silent["CRITICAL - effectively silent"]
    r0 --> rich["RichHandler - FluidLogRender"]
    log_all --> rich
    silent --> rich
    rich --> console["Rich Console - rank-aware theming"]
    style entry fill:#4a9eff,color:#fff
```

## Metrics Tracking with History

The `History` class accumulates per-step metrics (loss, learning rate, throughput,
etc.) and provides aggregation via `xarray`. It can write metrics to JSONL files,
log them to Weights & Biases, generate markdown reports, and render terminal
plots — all from the same recorded data.

```mermaid
flowchart LR
    subgraph training["Training Loop"]
        step["optimizer.step()"]
        record["history.update(metrics)"]
    end

    subgraph history_mod["ezpz.History"]
        store["Per-key metric lists"]
        stats["xarray statistics: mean, std, min, max"]
        report["Markdown report with embedded plots"]
    end

    subgraph outputs["Outputs"]
        jsonl["metrics.jsonl"]
        wandb_out["wandb logging"]
        md["report.md"]
        tplot_out["Terminal plots via ezpz.tplot"]
    end

    step --> record --> store
    store --> stats --> report
    store --> jsonl
    store --> wandb_out
    report --> md
    stats --> tplot_out
```

## Diagnostics (`ezpz doctor`)

Before submitting a multi-node job, `ezpz doctor` checks that all the
prerequisites are in place: MPI is installed and working, the scheduler is
detected, W&B credentials exist (or offline mode is set), and PyTorch can see
at least one accelerator. Each check returns a severity level (`ok`, `warning`,
`error`) and, when something is wrong, an actionable remediation hint.

```mermaid
flowchart LR
    doctor["ezpz doctor"]

    doctor --> c1["check_mpi - mpi4py + mpiexec"]
    doctor --> c2["check_scheduler - PBS / SLURM detection"]
    doctor --> c3["check_wandb - API key + .netrc"]
    doctor --> c4["check_torch_device - accelerator availability"]
    doctor --> c5["check_hostfile - consistency validation"]

    c1 --> result["CheckResult: ok / warning / error + remedy"]
    c2 --> result
    c3 --> result
    c4 --> result
    c5 --> result

    style doctor fill:#4a9eff,color:#fff
```

## Public API Surface

The public API is split into two namespaces: `ezpz` for the core distributed
training functions, and `ezpz.tp` for tensor parallelism. Most users only need
the `ezpz` namespace — `setup_torch()`, `wrap_model()`, and `cleanup()` cover
the common case. The `tp` namespace is for advanced users composing
multi-dimensional parallelism.

```mermaid
classDiagram
    class ezpz {
        +setup_torch(framework, backend, seed) int
        +cleanup()
        +wrap_model(model, use_fsdp, dtype) Module
        +wrap_model_for_ddp(model) DDP
        +wrap_model_for_fsdp(model, **kwargs) Module
        +wrap_model_for_fsdp2(model, **kwargs) Module
        +get_rank() int
        +get_local_rank() int
        +get_world_size(total, in_use) int
        +get_world_size_total() int
        +get_world_size_in_use() int
        +get_num_nodes() int
        +get_gpus_per_node() int
        +get_cpus_per_node() int
        +get_node_index() int
        +get_device_properties(device) dict
        +get_torch_device() str
        +get_torch_device_type() str
        +get_torch_backend() str
        +TORCH_DTYPES_MAP dict
        +synchronize()
        +barrier(group, implementation)
        +broadcast(obj, root) Any
        +all_reduce(obj, op, implementation) Any
        +get_dist_info() dict
        +get_hostname() str
        +get_machine() str
        +print_dist_setup() str
        +query_environment() dict
        +seed_everything(seed)
        +setup_wandb(project_name) Run
        +verify_wandb() bool
        +timeitlogit(rank) Callable
        +log_dict_as_bulleted_list(d, name)
        +get_nodes_from_hostfile(hostfile) list
        +get_hostfile_with_fallback(hostfile) Path
        +write_localhost_to_hostfile(hostfile)
        +write_hostfile_from_list_of_hosts(hosts, hostfile) Path
    }

    class tp {
        +initialize_tensor_parallel(tp, pp, cp)
        +destroy_tensor_parallel()
        +get_tensor_parallel_group() ProcessGroup
        +get_tensor_parallel_rank() int
        +get_tensor_parallel_world_size() int
        +get_data_parallel_group() ProcessGroup
        +get_data_parallel_rank() int
        +get_data_parallel_world_size() int
        +get_pipeline_parallel_group() ProcessGroup
        +get_context_parallel_group() ProcessGroup
    }

    ezpz --> tp : initialize via setup_torch
```

## Typical Usage Pattern

```python
# 1. Setup — single line bootstraps everything
import ezpz
rank = ezpz.setup_torch(seed=42)

# 2. Build model & wrap for distributed training
model = MyModel().to(ezpz.get_torch_device())
model = ezpz.wrap_model(model, use_fsdp=True, dtype="bf16")

# 3. Standard PyTorch training loop
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()

# 4. Cleanup
ezpz.cleanup()
```

```mermaid
flowchart LR
    A["import ezpz"] --> B["setup_torch()"]
    B --> C["wrap_model()"]
    C --> D["Training Loop"]
    D --> E["cleanup()"]

    style A fill:#95a5a6,color:#fff
    style B fill:#2ecc71,color:#fff
    style C fill:#3498db,color:#fff
    style D fill:#e67e22,color:#fff
    style E fill:#e74c3c,color:#fff
```

Launch with:

```bash
ezpz launch -- python train.py
```

## Project Structure

```
ezpz/
  src/ezpz/
    __init__.py           Lazy-loading package entry
    distributed.py        Core distributed API (clean rewrite)
    dist.py               Legacy distributed module
    configs.py            Configuration & constants
    jobs.py               Job scheduler metadata
    launch.py             Cross-node launcher
    slurm.py / pbs.py     Scheduler-specific utilities
    model.py / train.py   Model setup & training smoke tests
    history.py            Metrics tracking & aggregation
    profile.py            Performance profiling
    doctor.py             Runtime diagnostics
    lazy.py               Lazy import utilities
    integrations.py       WandB & HuggingFace integrations
    tp/                   Tensor parallelism (groups, layers, mappings)
    log/                  Rich-based rank-aware logging
    data/                 Distributed data loading (TP-aware)
    models/               LLaMA, ViT, minimal test models
    utils/                DeepSpeed configs, memory profiling
    cli/                  Click-based CLI (launch, test, doctor)
    examples/             FSDP, FSDP+TP, HF Trainer, ViT, diffusion
  tests/                  Test suite
  docs/                   MkDocs + Material documentation site
```
