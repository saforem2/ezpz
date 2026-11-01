# 🧰 `ezpz` CLI

Once installed, `ezpz` provides a CLI with a few useful utilities to help launch
distributed PyTorch applications.

Explicitly, these are `ezpz <command>`:

- 🚀 [`ezpz launch`](./launch/index.md):
  Launch commands with _automatic **job scheduler** detection_ (PBS, Slurm)
    - 💯 [`ezpz test`](./test.md):
      Run simple distributed smoke test[^wrapper].
    - 📊 [`ezpz benchmark`](./benchmark.md):
      Run all examples and generate a report
    - 📮 [`ezpz submit`](./submit.md):
      Submit jobs to PBS (`qsub`) or SLURM (`sbatch`); generates job scripts
      automatically
- 📊 [`ezpz status`](./status.md):
  Inspect job status locally or over SSH/IRI (fanout, filters, JSON)
- 🚫 [`ezpz cancel`](./cancel.md):
  Cancel a queued or running job (local or remote)
- ⏱️ [`ezpz extend`](./extend.md) / [`ezpz update-job`](./extend.md):
  Extend or retarget a queued job — walltime, queue, account
- 📄 [`ezpz fetch`](./fetch.md):
  Read a job's stdout without opening an interactive SSH session
- 📝 [`ezpz logs`](./logs.md):
  Browse per-example benchmark logs for a run
- 🖥️ [`ezpz resources`](./resources.md):
  List IRI compute resources (Aurora, Polaris, Crux, …)
- 🚨 [`ezpz incidents`](./incidents.md):
  List IRI incidents — maintenance windows, outages, facility events
- 📦 [`ezpz yeet`](./yeet.md):
  Distribute files (envs, models, datasets, etc.) to all worker nodes via
  parallel rsync
    - 🗜️ [`ezpz tar-env`](./tar-env.md):
      Package current Python environment as a tarball
- 🩺 [`ezpz doctor`](./doctor.md):
  Health check your environment
- 💀 [`ezpz kill`](./kill.md):
  Kill ezpz-launched python processes (local node or `--all-nodes`)
- ⬆️ [`ezpz update`](./update.md):
  Upgrade `ezpz` in the active environment and refresh `utils.sh`
- 📤 [`ezpz export-amsc`](./export-amsc.md):
  Export a finished run as one AmSC at-scale benchmarks CSV row
- ⌨️ [`ezpz completion`](./completion.md):
  Emit shell completion for `bash`/`zsh`/`fish`
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
        benchmark    Run all ezpz examples sequentially and generate a report.
        cancel       Cancel a queued or running job.
        completion   Emit shell completion for bash/zsh/fish.
        doctor       Inspect the environment for ezpz launch readiness.
        export-amsc  Export a finished run as one AmSC at-scale benchmarks row.
        extend       Extend or retarget a queued job (IRI PUT / SSH qalter).
        fetch        Read a job's stdout without opening an interactive SSH session.
        incidents    List IRI incidents — maintenance windows, outages, facility events.
        kill         Kill ezpz-launched python processes (or any matching pattern).
        launch       Launch a command across the active scheduler.
        logs         Browse per-example benchmark logs.
        resources    List IRI compute resources.
        status       Show job status (local or remote).
        submit       Submit a job to the active scheduler (PBS/SLURM).
        tar-env      Create (or locate) a tarball for the current environment.
        test         Run the distributed smoke test.
        update       Upgrade ezpz in the active environment and refresh utils.sh.
        update-job   Alias for extend.
        yeet         Distribute files (envs, models, datasets, etc.) to worker nodes.
        ```

[^wrapper]: This is really just a wrapper around:

    ```bash
    ezpz launch python3 -m ezpz.examples.test
    ```
