# 🧰 `ezpz` CLI

Once installed, `ezpz` provides a CLI with a few useful utilities
to help with launch distributed PyTorch applications.

Explicitly, these are `ezpz <command>`:

- [`ezpz doctor`](./doctor.md): Health check your environment
- [`ezpz test`](./test.md): Run simple distributed smoke test
- [`ezpz launch`](./launch/index.md): Launch arbitrary distributed commands  
   with _automatic **job scheduler** detection_ (PBS, Slurm) !!

To see the list of available commands, run:

```bash
ezpz --help
```

<!-- /// note | CLI: `ezpz` -->
<!---->
<!-- Once installed, `ezpz` provides a CLI with a few useful utilities -->
<!-- to help with distributed launches and environment validation. -->
<!---->
<!-- Explicitly, these are: -->
<!---->
<!-- ```bash -->
<!-- ezpz doctor  # environment validation and health-check -->
<!-- ezpz test    # distributed smoke test -->
<!-- ezpz launch  # general purpose, scheduler-aware launching -->
<!-- ``` -->
<!---->
<!-- To see the list of available commands, run: -->
<!---->
<!-- ```bash -->
<!-- ezpz --help -->
<!-- ``` -->
<!---->
<!-- /// note | 🧰 CLI Toolbox -->
<!---->
<!-- Checkout [🧰 **CLI**](https://saforem2.github.io/ezpz/cli/) for additional information. -->
<!---->
<!-- /// -->
<!---->
<!-- /// -->
<!---->
