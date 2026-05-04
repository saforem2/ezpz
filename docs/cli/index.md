# 🧰 `ezpz` CLI

Once installed, `ezpz` provides a CLI with a few useful utilities to help launch
distributed PyTorch applications.

Explicitly, these are `ezpz <command>`:

- 🚀 [`ezpz launch`](./launch/index.md):
  Launch commands with _automatic **job scheduler** detection_ (PBS, Slurm)
    - 💯 [`ezpz test`](./test.md):
      Run simple distributed smoke test[^wrapper].
    - 📊 [`ezpz benchmark`](./benchmark.md):
      Run all examples and generate a report
    - 📮 [`ezpz submit`](./submit.md):
      Submit jobs to PBS (`qsub`) or SLURM (`sbatch`); generates job scripts
      automatically
- 📦 [`ezpz yeet`](./yeet.md):
  Distribute files (envs, models, datasets, etc.) to all worker nodes via
  parallel rsync
    - 🗜️ [`ezpz tar-env`](./tar-env.md):
      Package current Python environment as a tarball
- 🩺 [`ezpz doctor`](./doctor.md):
  Health check your environment
- 💀 [`ezpz kill`](./kill.md):
  Kill ezpz-launched python processes (local node or `--all-nodes`)
- 📝 [`ezpz.examples`](../examples/index.md):
  Collection of distributed training examples (DDP, FSDP, ViT, FSDP+TP,
  diffusion, HF, HF Trainer, inference)

- ??? tip "`ezpz --help`"

        To see the list of available commands, run:

        ```bash
        $ ezpz --help
        Usage: ezpz [OPTIONS] COMMAND [ARGS]...

        ezpz distributed utilities.

        Options:
        --version   Show the version and exit.
        -h, --help  Show this message and exit.

        Commands:
        benchmark  Run all ezpz examples sequentially and generate a report.
        doctor     Inspect the environment for ezpz launch readiness.
        kill       Kill ezpz-launched python processes (or any matching pattern).
        launch     Launch a command across the active scheduler.
        submit     Submit a job to the active scheduler (PBS/SLURM).
        tar-env    Create (or locate) a tarball for the current environment.
        test       Run the distributed smoke test.
        yeet       Distribute files (envs, models, datasets, etc.) to worker nodes.
        ```

[^wrapper]: This is really just a wrapper around:

    ```bash
    ezpz launch python3 -m ezpz.examples.test
    ```
