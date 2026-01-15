# 🧰 `ezpz` CLI

Once installed, `ezpz` provides a CLI with a few useful utilities
to help launch distributed PyTorch applications.

Explicitly, these are `ezpz <command>`:

- 🩺 [`ezpz doctor`](./doctor.md): Health check your environment
- 🚀 [`ezpz launch`](./launch/index.md): Launch commands with _automatic
  **job scheduler** detection_ (PBS, Slurm)
    - 💯 [`ezpz test`](./test.md): Run simple distributed smoke test
    - 📝 [`ezpz.examples.*`](../examples/index.md): Scalable and _ready-to-go_!

        | Links                                                                                                                                                                                                                    | Example Module             | What it Does                                    |
        | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- | ----------------------------------------------- |
        | [:lucide-book:][ex-test-dist] · [:lucide-file-code:][api-test-dist] · [:lucide-github:][gh-test-dist]                                                                                                                    | `ezpz.examples.test_dist`  | Train MLP with DDP on MNIST                     |
        | [:lucide-book:][ex-fsdp] · [:lucide-file-code:][api-fsdp] · [:lucide-github:][gh-fsdp]                                                                                                                                   | `ezpz.examples.fsdp`       | Train CNN with FSDP on MNIST                    |
        | [:lucide-book:][ex-vit] · [:lucide-file-code:][api-vit] · [:lucide-github:][gh-vit]                            | `ezpz.examples.vit`        | Train ViT with FSDP on MNIST                    |
        | [:lucide-book:][ex-fsdp-tp] · [:lucide-file-code:][api-fsdp-tp] · [:lucide-github:][gh-fsdp-tp]                | `ezpz.examples.fsdp_tp`    | Train Transformer with FSDP + TP on HF Datasets |
        | [:lucide-book:][ex-diffusion] · [:lucide-file-code:][api-diffusion] · [:lucide-github:][gh-diffusion]          | `ezpz.examples.diffusion`  | Train Diffusion LLM with FSDP on HF Datasets    |
        | [:lucide-book:][ex-hf-trainer] · [:lucide-file-code:][api-hf-trainer] · [:lucide-github:][gh-hf-trainer] | `ezpz.examples.hf_trainer` | Train LLM with FSDP + HF Trainer on HF Datasets |


  [ex-test-dist]: ../examples/test-dist.md "Example"
  [api-test-dist]: ../python/Code-Reference/test_dist.md "API Reference"
  [gh-test-dist]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py "GitHub Source"
  [ex-fsdp]: ../examples/fsdp.md "Example"
  [api-fsdp]: ../python/Code-Reference/examples/fsdp.md "API Reference"
  [gh-fsdp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py "GitHub Source"
  [ex-vit]: ../examples/vit.md "Example"
  [api-vit]: ../python/Code-Reference/examples/vit.md "API Reference"
  [gh-vit]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py "GitHub Source"
  [ex-fsdp-tp]: ../examples/fsdp-tp.md "Example"
  [api-fsdp-tp]: ../python/Code-Reference/examples/fsdp_tp.md "API Reference"
  [gh-fsdp-tp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py "GitHub Source"
  [ex-diffusion]: ../examples/diffusion.md "Example"
  [api-diffusion]: ../python/Code-Reference/examples/diffusion.md "API Reference"
  [gh-diffusion]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py "GitHub Source"
  [ex-hf-trainer]: ../examples/hf-trainer/index.md "Example"
  [api-hf-trainer]: ../python/Code-Reference/examples/hf_trainer.md "API Reference"
  [gh-hf-trainer]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py "GitHub Source"


- ??? tip "`ezpz --help`"

        To see the list of available commands, run:

        ```shell-session
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
<!-- Checkout [🧰 **CLI**](https://ezpz.cool/cli/) for additional information. -->
<!---->
<!-- /// -->
<!---->
<!-- /// -->
<!---->
