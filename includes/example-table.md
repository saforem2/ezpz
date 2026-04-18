1. 📝 [**Examples**](https://ezpz.cool/examples/): Scalable and _ready-to-go_!

    | Links                                                                                                                                                                                                                    | Example Module             | What it Does                                    |
    | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- | ----------------------------------------------- |
    | [:lucide-book:][ex-test] · [:lucide-file-code:][api-test] · [:fontawesome-brands-github:][gh-test]                                                                                                                    | `ezpz.examples.test`  | Train MLP with DDP on MNIST                     |
    | [:lucide-book:][ex-fsdp] · [:lucide-file-code:][api-fsdp] · [:fontawesome-brands-github:][gh-fsdp]                                                                                                                                   | `ezpz.examples.fsdp`       | Train CNN with FSDP on MNIST                    |
    | [:lucide-book:][ex-vit] · [:lucide-file-code:][api-vit] · [:fontawesome-brands-github:][gh-vit]                            | `ezpz.examples.vit`        | Train ViT with FSDP on MNIST                    |
    | [:lucide-book:][ex-fsdp-tp] · [:lucide-file-code:][api-fsdp-tp] · [:fontawesome-brands-github:][gh-fsdp-tp]                | `ezpz.examples.fsdp_tp`    | Train Transformer with FSDP + TP on HF Datasets |
    | [:lucide-book:][ex-diffusion] · [:lucide-file-code:][api-diffusion] · [:fontawesome-brands-github:][gh-diffusion]          | `ezpz.examples.diffusion`  | Train Diffusion LLM with FSDP on HF Datasets    |
    | [:lucide-book:][ex-hf] · [:lucide-file-code:][api-hf] · [:fontawesome-brands-github:][gh-hf]                               | `ezpz.examples.hf`         | Fine-tune causal LM with Accelerate + FSDP       |
    | [:lucide-book:][ex-hf-trainer] · [:lucide-file-code:][api-hf-trainer] · [:fontawesome-brands-github:][gh-hf-trainer] | `ezpz.examples.hf_trainer` | Train LLM with FSDP + HF Trainer on HF Datasets |


    Any of the examples can be launched with:

    ```bash
    ezpz launch python3 -m ezpz.examples.<example>
    ```

    ??? tip "🤗 HF Integration"

        1. `ezpz.examples.`{[`fsdp_tp`](https://ezpz.cool/examples/fsdp-tp.md),
            [`diffusion`](https://ezpz.cool/examples/diffusion.md),
            [`hf`](https://ezpz.cool/examples/hf.md),
            [`hf_trainer`](https://ezpz.cool/examples/hf-trainer/index.md)}
            all support arbitrary 🤗 Hugging Face
            [datasets](https://huggingface.co/docs/datasets/index) e.g.:

            ```bash
            dataset="stanfordnlp/imdb"  # or any other HF dataset
            ezpz launch python3 -m ezpz.examples.fsdp_tp --dataset "${dataset}"
            ezpz launch python3 -m ezpz.examples.diffusion --dataset "${dataset}"
            ezpz launch python3 -m ezpz.examples.hf \
                --model_name_or_path meta-llama/Llama-3.2-1B \
                --dataset_name="${dataset}" \
                --streaming \
                --bf16=true
            ezpz launch python3 -m ezpz.examples.hf_trainer \
                --model_name_or_path meta-llama/Llama-3.2-1B \
                --dataset_name="${dataset}" \
                --streaming \
                --bf16=true
            ```

        1. [`ezpz.examples.hf`](https://ezpz.cool/examples/hf.md) and
            [`ezpz.examples.hf_trainer`](https://ezpz.cool/examples/hf-trainer/index.md) both support
            arbitrary combinations of (compatible) `transformers.from_pretrained`
            models, and HF Datasets (with support for streaming!).
            `hf` uses an explicit training loop with Accelerate, while
            `hf_trainer` wraps the HF `Trainer` API.

            ```bash
            ezpz launch python3 -m ezpz.examples.hf \
                --streaming \
                --dataset_name=eliplutchok/fineweb-small-sample \
                --tokenizer_name meta-llama/Llama-3.2-1B \
                --model_name_or_path meta-llama/Llama-3.2-1B \
                --bf16=true

            ezpz launch python3 -m ezpz.examples.hf_trainer \
                --streaming \
                --dataset_name=eliplutchok/fineweb-small-sample \
                --tokenizer_name meta-llama/Llama-3.2-1B \
                --model_name_or_path meta-llama/Llama-3.2-1B \
                --bf16=true
            ```

    ??? example "Simple Example"

        ```bash
        ezpz launch python3 -c 'import ezpz; print(ezpz.setup_torch())'
        ```

        ??? abstract "Output"

            ??? success "Macbook Pro"

                ```bash
                #[01/08/26 @ 14:56:50][~/v/s/ezpz][dev][$✘!?] [4s]
                ; ezpz launch python3 -c 'import ezpz; print(ezpz.setup_torch())'
                [2026-01-08 14:56:54,307030][I][ezpz/launch:515:run] No active scheduler detected; falling back to local mpirun: mpirun -np 2 python3 -c 'import ezpz; print(ezpz.setup_torch())'
                Using [2 / 2] available "mps" devices !!
                0
                1
                [2025-12-23-162222] Execution time: 4s sec
                ```

            ??? success "Aurora (2 Nodes)"

                ```bash
                #[aurora_frameworks-2025.2.0](torchtitan-aurora_frameworks-2025.2.0)[1m9s]
                #[01/08/26,14:56:42][x4418c6s1b0n0][/f/d/f/p/p/torchtitan][main][?]
                ; ezpz launch python3 -c 'import ezpz; print(ezpz.setup_torch())'


                [2026-01-08 14:58:01,994729][I][numexpr/utils:148:_init_num_threads] Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
                [2026-01-08 14:58:01,997067][I][numexpr/utils:151:_init_num_threads] Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
                [2026-01-08 14:58:01,997545][I][numexpr/utils:164:_init_num_threads] NumExpr defaulting to 16 threads.
                [2026-01-08 14:58:02,465850][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2026-01-08-145802]----
                [2026-01-08 14:58:04,765720][I][ezpz/launch:416:launch] Job ID: 8247203
                [2026-01-08 14:58:04,766527][I][ezpz/launch:417:launch] nodelist: ['x4418c6s1b0n0', 'x4717c0s6b0n0']
                [2026-01-08 14:58:04,766930][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                [2026-01-08 14:58:04,767616][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
                [2026-01-08 14:58:04,768399][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
                [2026-01-08 14:58:04,768802][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
                [2026-01-08 14:58:04,769517][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 -c 'import ezpz; print(ezpz.setup_torch())'
                [2026-01-08 14:58:04,770278][I][ezpz/launch:433:launch] Took: 3.01 seconds to build command.
                [2026-01-08 14:58:04,770660][I][ezpz/launch:436:launch] Executing:
                mpiexec
                --envall
                --np=24
                --ppn=12
                --hostfile=/var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                --no-vni
                --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
                python3
                -c
                import ezpz; print(ezpz.setup_torch())
                [2026-01-08 14:58:04,772125][I][ezpz/launch:220:get_aurora_filters] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG
                [2026-01-08 14:58:04,772651][I][ezpz/launch:443:launch] Execution started @ 2026-01-08-145804...
                [2026-01-08 14:58:04,773070][I][ezpz/launch:138:run_command] Caught 24 filters
                [2026-01-08 14:58:04,773429][I][ezpz/launch:139:run_command] Running command:
                mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 -c 'import ezpz; print(ezpz.setup_torch())'
                ```

                <details><summary>CPU bind output (24 lines)</summary>

                ```
                cpubind:list x4717c0s6b0n0 pid 118589 rank 12 0: mask 0x1c
                cpubind:list x4717c0s6b0n0 pid 118590 rank 13 1: mask 0x1c00
                ...
                cpubind:list x4418c6s1b0n0 pid 66460 rank 10 10: mask 0x1c000000000000000000000
                cpubind:list x4418c6s1b0n0 pid 66461 rank 11 11: mask 0x1c00000000000000000000000
                ```

                </details>

                ```bash
                Using [24 / 24] available "xpu" devices !!
                ```

                <details><summary>Raw rank output (24 lines)</summary>

                ```
                8
                10
                0
                4
                ...
                18
                21
                ```

                </details>

                ```bash
                [2026-01-08 14:58:14,252433][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2026-01-08-145814]----
                [2026-01-08 14:58:14,253726][I][ezpz/launch:448:launch] Execution finished with 0.
                [2026-01-08 14:58:14,254184][I][ezpz/launch:449:launch] Executing finished in 9.48 seconds.
                [2026-01-08 14:58:14,254555][I][ezpz/launch:450:launch] Took 9.48 seconds to run. Exiting.
                took: 18s
                ```

    ??? example "`demo.py`"

        ```python title="demo.py"
        import ezpz

        # automatic device + backend setup for distributed PyTorch
        _ = ezpz.setup_torch()  # CUDA/NCCL, XPU/XCCL, {MPS, CPU}/GLOO, ...

        device = ezpz.get_torch_device() # {cuda, xpu, mps, cpu, ...}
        rank = ezpz.get_rank()
        world_size = ezpz.get_world_size()
        # ...etc

        if rank == 0:
            print(f"Hello from rank {rank} / {world_size} on {device}!")
        ```

        We can launch this script with:

        ```bash
        ezpz launch python3 demo.py
        ```

        ??? abstract "Output(s)"

            ??? success "MacBook Pro"

                ```bash
                # from MacBook Pro
                $ ezpz launch python3 demo.py
                [2026-01-08 07:22:31,989741][I][ezpz/launch:515:run] No active scheduler detected; falling back to local mpirun: mpirun -np 2 python3 /Users/samforeman/python/ezpz_demo.py
                Using [2 / 2] available "mps" devices !!
                Hello from rank 0 / 2 on mps!
                ```

            ??? success "Aurora (2 nodes)"

                ```bash
                # from 2 nodes of Aurora:
                #[aurora_frameworks-2025.2.0](foremans-aurora_frameworks-2025.2.0)[C v7.5.0-gcc][43s]
                #[01/08/26,07:26:10][x4604c5s2b0n0][~]
                ; ezpz launch python3 demo.py

                [2026-01-08 07:26:19,723138][I][numexpr/utils:148:_init_num_threads] Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
                [2026-01-08 07:26:19,725453][I][numexpr/utils:151:_init_num_threads] Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
                [2026-01-08 07:26:19,725932][I][numexpr/utils:164:_init_num_threads] NumExpr defaulting to 16 threads.
                [2026-01-08 07:26:20,290222][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2026-01-08-072620]----
                [2026-01-08 07:26:21,566797][I][ezpz/launch:416:launch] Job ID: 8246832
                [2026-01-08 07:26:21,567684][I][ezpz/launch:417:launch] nodelist: ['x4604c5s2b0n0', 'x4604c5s3b0n0']
                [2026-01-08 07:26:21,568082][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                [2026-01-08 07:26:21,568770][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
                [2026-01-08 07:26:21,569557][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
                [2026-01-08 07:26:21,569959][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
                [2026-01-08 07:26:21,570821][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 demo.py
                [2026-01-08 07:26:21,571548][I][ezpz/launch:433:launch] Took: 2.11 seconds to build command.
                [2026-01-08 07:26:21,571918][I][ezpz/launch:436:launch] Executing:
                mpiexec
                --envall
                --np=24
                --ppn=12
                --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                --no-vni
                --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
                python3
                demo.py
                [2026-01-08 07:26:21,573262][I][ezpz/launch:220:get_aurora_filters] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG
                [2026-01-08 07:26:21,573781][I][ezpz/launch:443:launch] Execution started @ 2026-01-08-072621...
                [2026-01-08 07:26:21,574195][I][ezpz/launch:138:run_command] Caught 24 filters
                [2026-01-08 07:26:21,574532][I][ezpz/launch:139:run_command] Running command:
                mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 demo.py
                ```

                <details><summary>CPU bind output (24 lines)</summary>

                ```
                cpubind:list x4604c5s3b0n0 pid 131587 rank 12 0: mask 0x1c
                cpubind:list x4604c5s3b0n0 pid 131588 rank 13 1: mask 0x1c00
                ...
                cpubind:list x4604c5s2b0n0 pid 121235 rank 10 10: mask 0x1c000000000000000000000
                cpubind:list x4604c5s2b0n0 pid 121236 rank 11 11: mask 0x1c00000000000000000000000
                ```

                </details>

                ```bash
                Using [24 / 24] available "xpu" devices !!
                Hello from rank 0 / 24 on xpu!
                [2026-01-08 07:26:33,060432][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2026-01-08-072633]----
                [2026-01-08 07:26:33,061512][I][ezpz/launch:448:launch] Execution finished with 0.
                [2026-01-08 07:26:33,062045][I][ezpz/launch:449:launch] Executing finished in 11.49 seconds.
                [2026-01-08 07:26:33,062531][I][ezpz/launch:450:launch] Took 11.49 seconds to run. Exiting.
                took: 22s
                ```


  [ex-test]: examples/test.md "Example"
  [api-test]: python/Code-Reference/examples/test.md "API Reference"
  [gh-test]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/test.py "GitHub Source"
  [ex-fsdp]: examples/fsdp.md "Example"
  [api-fsdp]: python/Code-Reference/examples/fsdp.md "API Reference"
  [gh-fsdp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py "GitHub Source"
  [ex-vit]: examples/vit.md "Example"
  [api-vit]: python/Code-Reference/examples/vit.md "API Reference"
  [gh-vit]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py "GitHub Source"
  [ex-fsdp-tp]: examples/fsdp-tp.md "Example"
  [api-fsdp-tp]: python/Code-Reference/examples/fsdp_tp.md "API Reference"
  [gh-fsdp-tp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py "GitHub Source"
  [ex-diffusion]: examples/diffusion.md "Example"
  [api-diffusion]: python/Code-Reference/examples/diffusion.md "API Reference"
  [gh-diffusion]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py "GitHub Source"
  [ex-hf]: examples/hf.md "Example"
  [api-hf]: python/Code-Reference/examples/hf.md "API Reference"
  [gh-hf]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf.py "GitHub Source"
  [ex-hf-trainer]: examples/hf-trainer/index.md "Example"
  [api-hf-trainer]: python/Code-Reference/examples/hf_trainer.md "API Reference"
  [gh-hf-trainer]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py "GitHub Source"
