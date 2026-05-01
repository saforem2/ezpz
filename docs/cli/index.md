# 🧰 `ezpz` CLI

Once installed, `ezpz` provides a CLI with a few useful utilities
to help launch distributed PyTorch applications.

Explicitly, these are `ezpz <command>`:

- 🚀 [`ezpz launch`](./launch/index.md): Launch commands with _automatic
  **job scheduler** detection_ (PBS, Slurm)
- 📮 [`ezpz submit`](./submit.md): Submit jobs to PBS (`qsub`) or SLURM
  (`sbatch`) — generates job scripts automatically
- 💯 [`ezpz test`](./test.md): Run simple distributed smoke test
- 📊 [`ezpz benchmark`](./benchmark.md): Run all examples and generate a
  report
- 🩺 [`ezpz doctor`](./doctor.md): Health check your environment
- 📝 [`ezpz.examples`](../examples/index.md): Collection of distributed
  training examples
    - ??? note "Distributed Training Examples"

            See the [Examples](../examples/index.md) page for full details.

            - [`test`](../examples/test.md): Simplest DDP training loop

                ```bash
                ezpz launch python3 -m ezpz.examples.test
                ```

            - [`fsdp`](../examples/fsdp.md): FSDP for memory-efficient training

                ```bash
                ezpz launch python3 -m ezpz.examples.fsdp
                ```

            - [`vit`](../examples/vit.md): Vision Transformer with FSDP + optional `torch.compile`

                ```bash
                ezpz launch python3 -m ezpz.examples.vit
                ```

            - [`fsdp_tp`](../examples/fsdp-tp.md): 2D parallelism (FSDP + Tensor Parallel)

                ```bash
                ezpz launch python3 -m ezpz.examples.fsdp_tp
                ```

            - [`diffusion`](../examples/diffusion.md): Diffusion model training with FSDP

                ```bash
                ezpz launch python3 -m ezpz.examples.diffusion
                ```

            - [`hf`](../examples/hf.md): Fine-tune causal LM with explicit training loop (Accelerate + FSDP)

                ```bash
                ezpz launch python3 -m ezpz.examples.hf
                ```

            - [`hf_trainer`](../examples/hf-trainer/index.md): Hugging Face Trainer integration

                ```bash
                ezpz launch python3 -m ezpz.examples.hf_trainer
                ```

- ??? question "Experimental"

        - 📦 [`ezpz tar-env`](./tar-env.md): Package current Python environment as a tarball
        - 🚀 [`ezpz yeet`](./yeet.md): Distribute files (envs, models, datasets, etc.) to all worker nodes via parallel rsync

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
        launch     Launch a command across the active scheduler.
        submit     Submit a job to the active scheduler (PBS/SLURM).
        tar-env    Create (or locate) a tarball for the current environment.
        test       Run the distributed smoke test.
        yeet       Distribute files (envs, models, datasets, etc.) to worker nodes.
        ```
