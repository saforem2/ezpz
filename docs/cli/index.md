# 🧰 `ezpz` CLI

Once installed, `ezpz` provides a CLI with a few useful utilities
to help launch distributed PyTorch applications.

Explicitly, these are `ezpz <command>`:

- 🩺 [`ezpz doctor`](./doctor.md): Health check your environment
- 🚀 [`ezpz launch`](./launch/index.md): Launch commands with _automatic
  **job scheduler** detection_ (PBS, Slurm)
    - 💯 [`ezpz test`](./test.md): Run simple distributed smoke test
    --8<-- "../includes/example-table.md"

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
        doctor    Inspect the environment for ezpz launch readiness.
        launch    Launch a command across the active scheduler.
        test      Run the distributed smoke test.
        ```

