# 🧰 `ezpz` CLI

Once installed, `ezpz` provides a CLI with a few useful utilities to help launch
distributed PyTorch applications.

Explicitly, these are `ezpz <command>`:

- 🚀 [`ezpz\u00a0launch`](./launch/index.md):
  Launch commands with _automatic **job scheduler** detection_ (PBS, Slurm)
    - 💯 [`ezpz\u00a0test`](./test.md):
      Run simple distributed smoke test[^wrapper].
    - 📊 [`ezpz\u00a0benchmark`](./benchmark.md):
      Run all examples and generate a report
    - 📮 [`ezpz\u00a0submit`](./submit.md):
      Submit jobs to PBS (`qsub`) or SLURM (`sbatch`); generates job scripts
      automatically
- 📊 [`ezpz\u00a0status`](./status.md):
  Inspect job status locally or over SSH/IRI (fanout, filters, JSON)
- 🚫 [`ezpz\u00a0cancel`](./cancel.md):
  Cancel a queued or running job (local or remote)
- ⏱️ [`ezpz\u00a0extend`](./extend.md) / [`ezpz\u00a0update-job`](./extend.md):
  Extend or retarget a queued job — walltime, queue, account
- 📄 [`ezpz\u00a0fetch`](./fetch.md):
  Read a job's stdout without opening an interactive SSH session
- 📝 [`ezpz\u00a0logs`](./logs.md):
  Browse per-example benchmark logs for a run
- 🖥️ [`ezpz\u00a0resources`](./resources.md):
  List IRI compute resources (Aurora, Polaris, Crux, …)
- 🚨 [`ezpz\u00a0incidents`](./incidents.md):
  List IRI incidents — maintenance windows, outages, facility events
- 📦 [`ezpz\u00a0yeet`](./yeet.md):
  Distribute files (envs, models, datasets, etc.) to all worker nodes via
  parallel rsync
    - 🗜️ [`ezpz\u00a0tar-env`](./tar-env.md):
      Package current Python environment as a tarball
- 🩺 [`ezpz\u00a0doctor`](./doctor.md):
  Health check your environment
- 💀 [`ezpz\u00a0kill`](./kill.md):
  Kill ezpz-launched python processes (local node or `--all-nodes`)
- ⬆️ [`ezpz\u00a0update`](./update.md):
  Upgrade `ezpz` in the active environment and refresh `utils.sh`
- 📤 [`ezpz\u00a0export-amsc`](./export-amsc.md):
  Export a finished run as one AmSC at-scale benchmarks CSV row
- ⌨️ [`ezpz\u00a0completion`](./completion.md):
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
