# 🚀 `ezpz launch`

Single entry point for launching distributed applications.

```bash
ezpz launch <cmd>
```

This will:

1. Automatically detect your PBS/Slurm job and 
2. Launch `<cmd>` across all available accelerators.

This is done by detecting if `ezpz launch` is being executed from inside a
PBS/Slurm job[^schedulers].

If so, it determines the specifics of the active job (number of nodes, and
number of GPUs _per_ node), and uses this information to build and execute the
appropriate launch command (e.g. `mpiexec`, `srun`).

When not running inside a PBS/Slurm job, `ezpz launch` falls back to `mpirun`
with sensible defaults.

Arguments can be passed through to the `mpiexec`/`srun` launcher by separating
them from the `<cmd>` with `--`[^launch-args], e.g.:

```bash
ezpz launch <launch-args> -- <cmd> <cmd-args>
```

[^launch-args]: When no `--` is present, all arguments are treated as part of
    the command to run.

For example, to run with 8 processes total, 4 processes per node, on 2 hosts,
we can:

```bash
ezpz launch -n 8 -ppn 4 -nh 2 -- python3 -m ezpz.examples.fsdp_tp
```

Assuming your current job can satisfy this (i.e. at least 4 accelerators per
node, and at least 2 nodes), this would launch `python3 -m
ezpz.examples.fsdp_tp` across 8 processes, 4 per node, on the first two hosts
allocated to your job.

- ??? abstract "`ezpz launch --help`"

        ```bash
        ezpz launch --help
        usage: ezpz launch [-h] [--print-source] [--filter FILTER [FILTER ...]] [-n NPROC] [-ppn NPROC_PER_NODE] [-nh NHOSTS] [--hostfile HOSTFILE] ...

        Launch a command on the current PBS/SLURM job.

        Additional `<launcher flags>` can be passed through directly
        to the launcher by including '--' as a separator before
        the command.

        Examples:

            ezpz launch <launcher flags> -- <command> <args>

            ezpz launch -n 8 -ppn 4 --verbose --tag-output -- python3 -m ezpz.examples.fsdp_tp

            ezpz launch --nproc 8 -x EZPZ_LOG_LEVEL=DEBUG -- python3 my_script.py --my-arg val

        positional arguments:
        command               Command (and arguments) to execute. Use '--' to separate options when needed.

        options:
        -h, --help            show this help message and exit
        --print-source        Print the location of the launch CLI source and exit.
        --filter FILTER [FILTER ...]
                                Filter output lines by these strings.
        -n NPROC, -np NPROC, --n NPROC, --np NPROC, --nproc NPROC, --world_size NPROC, --nprocs NPROC
                                Number of processes.
        -ppn NPROC_PER_NODE, --ppn NPROC_PER_NODE, --nproc_per_node NPROC_PER_NODE
                                Processes per node.
        -nh NHOSTS, --nh NHOSTS, --nhost NHOSTS, --nnode NHOSTS, --nnodes NHOSTS, --nhosts NHOSTS, --nhosts NHOSTS
                                Number of nodes to use.
        --hostfile HOSTFILE   Hostfile to use for launching.
        ```

## Python interpreter resolution

When `ezpz launch` needs to invoke `python3` (e.g. `ezpz launch -m my.module`),
it picks the interpreter in this order:

1. **`$VIRTUAL_ENV/bin/python3`** if `$VIRTUAL_ENV` is set and exists
2. **`shutil.which("python3")`** — first python3 on `$PATH`
3. **`sys.executable`** as a last resort

Why not just `sys.executable`? It's frozen at interpreter startup. If
you ran [`ezpz yeet-env`](../yeet-env.md) to copy your env to `/tmp/`
and then `source /tmp/.venv/bin/activate`, `sys.executable` would still
point to the original Lustre path because the `ezpz` CLI script's
shebang is baked in at install time. Reading `$VIRTUAL_ENV` (set by
`activate`) lets the launcher follow the user's actual current venv.

## Examples

Use it to launch:

- Arbitrary command(s):

    ```bash
    ezpz launch hostname
    ```

- Arbitrary Python string:

    ```bash
    ezpz launch python3 -c 'import ezpz; ezpz.setup_torch()'
    ```

- One of the Distributed Training examples:

    ```bash
    ezpz launch python3 -m ezpz.examples.test --profile
    ezpz launch -n 8 -- python3 -m ezpz.examples.fsdp_tp --tp 4
    ```

- Your own distributed training script:

    ```bash
    ezpz launch -n 16 -ppn 8 -- python3 -m your_app.train --config configs/your_config.yaml
    ```

    to launch `your_app.train` across 16 processes, 8 per node.

[^schedulers]: By default, this will detect if we're running behind a job
    scheduler (e.g. PBS or Slurm).<br>
    If so, we automatically determine the specifics of the currently active
    job; explicitly, this will determine:

    1. The number of available nodes
    2. How many GPUs are present on each of these nodes
    3. How many GPUs we have _total_

    It will then use this information to automatically construct the
    appropriate {`mpiexec`, `srun`} command to launch, and finally, execute the
    launch cmd.


??? tip "Sequence Diagram"

    Two primary control paths drive `ezpz launch`: a scheduler-aware path used when
    running inside PBS/SLURM allocations, and a local fallback that shells out to
    `mpirun` when no scheduler metadata is available.

    ```mermaid
    sequenceDiagram
        autonumber
        actor User
        participant CLI as ezpz_launch
        participant Scheduler as PBS_or_Slurm
        participant MPI as mpirun_mpiexec
        participant App as User_application

        User->>CLI: ezpz launch <launch_flags> -- <cmd> <cmd_flags>
        CLI->>Scheduler: detect_scheduler()
        alt scheduler_detected
            Scheduler-->>CLI: scheduler_type, job_metadata
            CLI->>Scheduler: build_scheduler_command(cmd_to_launch)
            Scheduler-->>CLI: launch_cmd (mpiexec_or_srun)
            CLI->>MPI: run_command(launch_cmd)
            MPI->>App: start_ranks_and_execute
            App-->>MPI: return_codes
            MPI-->>CLI: aggregate_status
        else no_scheduler_detected
            Scheduler-->>CLI: unknown
            CLI->>MPI: mpirun -np 2 <cmd> <cmd_flags>
            MPI->>App: start_local_ranks
            App-->>MPI: return_codes
            MPI-->>CLI: aggregate_status
        end
        CLI-->>User: exit_code
    ```

## Distributed Training Examples

--8<-- "../includes/example-table.md"


  [ex-test]: ../../examples/test.md "Example"
  [api-test]: ../../python/Code-Reference/examples/test.md "API Reference"
  [gh-test]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/test.py "GitHub Source"
  [ex-fsdp]: ../../examples/fsdp.md "Example"
  [api-fsdp]: ../../python/Code-Reference/examples/fsdp.md "API Reference"
  [gh-fsdp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py "GitHub Source"
  [ex-vit]: ../../examples/vit.md "Example"
  [api-vit]: ../../python/Code-Reference/examples/vit.md "API Reference"
  [gh-vit]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py "GitHub Source"
  [ex-fsdp-tp]: ../../examples/fsdp-tp.md "Example"
  [api-fsdp-tp]: ../../python/Code-Reference/examples/fsdp_tp.md "API Reference"
  [gh-fsdp-tp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py "GitHub Source"
  [ex-diffusion]: ../../examples/diffusion.md "Example"
  [api-diffusion]: ../../python/Code-Reference/examples/diffusion.md "API Reference"
  [gh-diffusion]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py "GitHub Source"
  [ex-hf-trainer]: ../../examples/hf-trainer/index.md "Example"
  [api-hf-trainer]: ../../python/Code-Reference/examples/hf_trainer.md "API Reference"
  [gh-hf-trainer]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py "GitHub Source"

[^distributed-history]: The `ezpz.History` class automatically computes
    distributed statistics (min, max, mean, std. dev) across ranks for all
    recorded metrics.  
    **NOTE**: This is automatically disabled when
    `ezpz.get_world_size() >= 384` (e.g. >= {32, 96} {Aurora, Polaris} nodes)
    due to the additional overhead introduced (but can be manually enabled, if
    desired).
