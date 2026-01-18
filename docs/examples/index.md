# 📝 Ready-to-go Examples

This section contains ready-to-run examples demonstrating various features of
the `ezpz` library.

- 📝 [`ezpz.examples.*`](../cli/index.md): Scalable and _ready-to-go_!

    | Links                                                                                                                                                                                                                    | Example Module             | What it Does                                    |
    | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- | ----------------------------------------------- |
    | [:lucide-book:][ex-test] · [:lucide-file-code:][api-test] · [:lucide-github:][gh-test]                                                                                                                    | `ezpz.examples.test`  | Train MLP with DDP on MNIST                     |
    | [:lucide-book:][ex-fsdp] · [:lucide-file-code:][api-fsdp] · [:lucide-github:][gh-fsdp]                                                                                                                                   | `ezpz.examples.fsdp`       | Train CNN with FSDP on MNIST                    |
    | [:lucide-book:][ex-vit] · [:lucide-file-code:][api-vit] · [:lucide-github:][gh-vit]                            | `ezpz.examples.vit`        | Train ViT with FSDP on MNIST                    |
    | [:lucide-book:][ex-fsdp-tp] · [:lucide-file-code:][api-fsdp-tp] · [:lucide-github:][gh-fsdp-tp]                | `ezpz.examples.fsdp_tp`    | Train Transformer with FSDP + TP on HF Datasets |
    | [:lucide-book:][ex-diffusion] · [:lucide-file-code:][api-diffusion] · [:lucide-github:][gh-diffusion]          | `ezpz.examples.diffusion`  | Train Diffusion LLM with FSDP on HF Datasets    |
    | [:lucide-book:][ex-hf-trainer] · [:lucide-file-code:][api-hf-trainer] · [:lucide-github:][gh-hf-trainer] | `ezpz.examples.hf_trainer` | Train LLM with FSDP + HF Trainer on HF Datasets |


    - ??? tip "Running Examples"

            Any of the examples below can be launched with (sensible defaults if not
            specified):

            ```shell-session
            ezpz launch python3 -m ezpz.examples.fsdp
            ezpz launch python3 -m ezpz.examples.fsdp_tp
            # ...etc
            ezpz launch python3 -m ezpz.examples.hf_trainer
            ```

    - ??? tip "🤗 HF Integration"

            1. `ezpz.examples.`{[`fsdp_tp`](./fsdp-tp.md),[`diffusion,hf_trainer`](./diffusion.md),[`hf_trainer`](./hf-trainer/index.md)} all support
                arbitrary 🤗 Hugging Face
                [datasets](https://huggingface.co/docs/datasets/index) e.g.:

                ```bash
                # use any --dataset from HF Datasets hub
                ezpz launch python3 -m ezpz.examples.fsdp_tp --dataset stanfordnlp/imdb
                ```

            1. [`ezpz.examples.hf_trainer`](./hf-trainer/index.md) supports
            arbitrary combinations of (compatible) `transformers.from_pretrained`
            models, and HF Datasets (with support for streaming!)

                ```bash
                ezpz launch python3 -m ezpz.examples.hf_trainer \
                    --streaming \
                    --dataset_name=eliplutchok/fineweb-small-sample \
                    --tokenizer_name meta-llama/Llama-3.2-1B \
                    --model_name_or_path meta-llama/Llama-3.2-1B \
                    --bf16=true
                    # ...etc.
                ```

    1. ??? example "`demo.py`"

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
                    cpubind:list x4604c5s3b0n0 pid 131587 rank 12 0: mask 0x1c
                    cpubind:list x4604c5s3b0n0 pid 131588 rank 13 1: mask 0x1c00
                    cpubind:list x4604c5s3b0n0 pid 131589 rank 14 2: mask 0x1c0000
                    cpubind:list x4604c5s3b0n0 pid 131590 rank 15 3: mask 0x1c000000
                    cpubind:list x4604c5s3b0n0 pid 131591 rank 16 4: mask 0x1c00000000
                    cpubind:list x4604c5s3b0n0 pid 131592 rank 17 5: mask 0x1c0000000000
                    cpubind:list x4604c5s3b0n0 pid 131593 rank 18 6: mask 0x1c0000000000000
                    cpubind:list x4604c5s3b0n0 pid 131594 rank 19 7: mask 0x1c000000000000000
                    cpubind:list x4604c5s3b0n0 pid 131595 rank 20 8: mask 0x1c00000000000000000
                    cpubind:list x4604c5s3b0n0 pid 131596 rank 21 9: mask 0x1c0000000000000000000
                    cpubind:list x4604c5s3b0n0 pid 131597 rank 22 10: mask 0x1c000000000000000000000
                    cpubind:list x4604c5s3b0n0 pid 131598 rank 23 11: mask 0x1c00000000000000000000000
                    cpubind:list x4604c5s2b0n0 pid 121225 rank 0 0: mask 0x1c
                    cpubind:list x4604c5s2b0n0 pid 121226 rank 1 1: mask 0x1c00
                    cpubind:list x4604c5s2b0n0 pid 121227 rank 2 2: mask 0x1c0000
                    cpubind:list x4604c5s2b0n0 pid 121228 rank 3 3: mask 0x1c000000
                    cpubind:list x4604c5s2b0n0 pid 121229 rank 4 4: mask 0x1c00000000
                    cpubind:list x4604c5s2b0n0 pid 121230 rank 5 5: mask 0x1c0000000000
                    cpubind:list x4604c5s2b0n0 pid 121231 rank 6 6: mask 0x1c0000000000000
                    cpubind:list x4604c5s2b0n0 pid 121232 rank 7 7: mask 0x1c000000000000000
                    cpubind:list x4604c5s2b0n0 pid 121233 rank 8 8: mask 0x1c00000000000000000
                    cpubind:list x4604c5s2b0n0 pid 121234 rank 9 9: mask 0x1c0000000000000000000
                    cpubind:list x4604c5s2b0n0 pid 121235 rank 10 10: mask 0x1c000000000000000000000
                    cpubind:list x4604c5s2b0n0 pid 121236 rank 11 11: mask 0x1c00000000000000000000000
                    Using [24 / 24] available "xpu" devices !!
                    Hello from rank 0 / 24 on xpu!
                    [2026-01-08 07:26:33,060432][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2026-01-08-072633]----
                    [2026-01-08 07:26:33,061512][I][ezpz/launch:448:launch] Execution finished with 0.
                    [2026-01-08 07:26:33,062045][I][ezpz/launch:449:launch] Executing finished in 11.49 seconds.
                    [2026-01-08 07:26:33,062531][I][ezpz/launch:450:launch] Took 11.49 seconds to run. Exiting.
                    took: 22s
                    ```

  [ex-test]: ./test.md "Example"
  [api-test]: ../python/Code-Reference/examples/test.md "API Reference"
  [gh-test]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/test.py "GitHub Source"
  [ex-fsdp]: ./fsdp.md "Example"
  [api-fsdp]: ../python/Code-Reference/examples/fsdp.md "API Reference"
  [gh-fsdp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py "GitHub Source"
  [ex-vit]: ./vit.md "Example"
  [api-vit]: ../python/Code-Reference/examples/vit.md "API Reference"
  [gh-vit]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py "GitHub Source"
  [ex-fsdp-tp]: ./fsdp-tp.md "Example"
  [api-fsdp-tp]: ../python/Code-Reference/examples/fsdp_tp.md "API Reference"
  [gh-fsdp-tp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py "GitHub Source"
  [ex-diffusion]: ./diffusion.md "Example"
  [api-diffusion]: ../python/Code-Reference/examples/diffusion.md "API Reference"
  [gh-diffusion]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py "GitHub Source"
  [ex-hf-trainer]: ./hf-trainer/index.md "Example"
  [api-hf-trainer]: ../python/Code-Reference/examples/hf_trainer.md "API Reference"
  [gh-hf-trainer]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py "GitHub Source"

<!-- 1. 📝 **Complete Examples**: `ezpz.examples.*`   -->
<!--     A collection of ready-to-go distributed training examples that can -->
<!--     be run at _**any scale**_, on **_any hardware_**: -->
<!--     - \[[:octicons-mark-github-16:](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/test.py)\] [`ezpz.examples.test`](../python/Code-Reference/examples/test.md): [Train MLP with DDP on MNIST](https://ezpz.cool/examples/test/) -->
<!--     - \[[:octicons-mark-github-16:](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py)\] [`ezpz.examples.fsdp`](../python/Code-Reference/examples/fsdp.md): [Train CNN with FSDP on MNIST](https://ezpz.cool/examples/fsdp/) -->
<!--     - \[[:octicons-mark-github-16:](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py)\] [`ezpz.examples.vit`](../python/Code-Reference/examples/vit.md): [Train ViT with FSDP on MNIST](https://ezpz.cool/examples/vit/) -->
<!--     - \[[:octicons-mark-github-16:](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py)\] [`ezpz.examples.fsdp_tp`](../python/Code-Reference/examples/fsdp_tp.md): [Train Transformer with FSDP and TP on HF Datasets](https://ezpz.cool/examples/fsdp-tp/) -->
<!--     - \[[:octicons-mark-github-16:](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py)\] [`ezpz.examples.diffusion`](../python/Code-Reference/examples/diffusion.md): [Train Diffusion LLM with FSDP on HF Datasets](https://ezpz.cool/examples/diffusion/) -->
<!--     - \[[:octicons-mark-github-16:](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py)\] [`ezpz.examples.hf_trainer`](../python/Code-Reference/examples/hf_trainer.md): [Train LLM with FSDP and HF Trainer on HF Datasets](https://ezpz.cool/examples/hf-trainer/) -->
<!---->
<!--     1. [`ezpz.examples.test`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/test.py): -->
<!--        [Train MLP with DDP on MNIST](https://ezpz.cool/examples/test/) -->
<!--     1. [`ezpz.examples.fsdp`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py): -->
<!--        [Train CNN with FSDP on MNIST](https://ezpz.cool/examples/fsdp/) -->
<!--     1. [`ezpz.examples.vit`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py): -->
<!--        [Train ViT with FSDP on MNIST](https://ezpz.cool/examples/vit/) -->
<!--     1. [`ezpz.examples.fsdp_tp`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py): -->
<!--        [Train Transformer with FSDP and TP on HF Datasets](https://ezpz.cool/examples/fsdp-tp/) -->
<!--     1. [`ezpz.examples.diffusion`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py): -->
<!--        [Train Diffusion LLM with FSDP on HF Datasets](https://ezpz.cool/examples/diffusion/) -->
<!--     1. [`ezpz.examples.hf_trainer`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py): -->
<!--        [Train LLM with FSDP and HF Trainer on HF Datasets](https://ezpz.cool/examples/hf-trainer/) -->
<!---->
<!--     1. ??? example "`demo.py`" -->
<!---->
<!--             ```python -->
<!--             # demo.py -->
<!--             import ezpz -->
<!---->
<!--             # automatic device + backend setup for distributed PyTorch -->
<!--             _ = ezpz.setup_torch()  # CUDA/NCCL, XPU/XCCL, {MPS, CPU}/GLOO, ... -->
<!---->
<!--             device = ezpz.get_torch_device() # {cuda, xpu, mps, cpu, ...} -->
<!--             rank = ezpz.get_rank() -->
<!--             world_size = ezpz.get_world_size() -->
<!--             # ...etc -->
<!---->
<!--             if rank == 0: -->
<!--                 print(f"Hello from rank {rank} / {world_size} on {device}!") -->
<!--             ``` -->
<!---->
<!--             We can launch this script with: -->
<!---->
<!--             ```bash -->
<!--             ezpz launch python3 demo.py -->
<!--             ``` -->
<!---->
<!--             ??? success "Output(s)" -->
<!---->
<!--                 ??? abstract "MacBook Pro" -->
<!---->
<!--                     ```bash -->
<!--                     # from MacBook Pro -->
<!--                     $ ezpz launch python3 demo.py -->
<!--                     [2026-01-08 07:22:31,989741][I][ezpz/launch:515:run] No active scheduler detected; falling back to local mpirun: mpirun -np 2 python3 /Users/samforeman/python/ezpz_demo.py -->
<!--                     Using [2 / 2] available "mps" devices !! -->
<!--                     Hello from rank 0 / 2 on mps! -->
<!--                     ``` -->
<!---->
<!--                 ??? abstract "Aurora (2 nodes)" -->
<!---->
<!--                     ```bash -->
<!--                     # from 2 nodes of Aurora: -->
<!--                     #[aurora_frameworks-2025.2.0](foremans-aurora_frameworks-2025.2.0)[C v7.5.0-gcc][43s] -->
<!--                     #[01/08/26,07:26:10][x4604c5s2b0n0][~] -->
<!--                     ; ezpz launch python3 demo.py -->
<!---->
<!--                     [2026-01-08 07:26:19,723138][I][numexpr/utils:148:_init_num_threads] Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable. -->
<!--                     [2026-01-08 07:26:19,725453][I][numexpr/utils:151:_init_num_threads] Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16. -->
<!--                     [2026-01-08 07:26:19,725932][I][numexpr/utils:164:_init_num_threads] NumExpr defaulting to 16 threads. -->
<!--                     [2026-01-08 07:26:20,290222][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2026-01-08-072620]---- -->
<!--                     [2026-01-08 07:26:21,566797][I][ezpz/launch:416:launch] Job ID: 8246832 -->
<!--                     [2026-01-08 07:26:21,567684][I][ezpz/launch:417:launch] nodelist: ['x4604c5s2b0n0', 'x4604c5s3b0n0'] -->
<!--                     [2026-01-08 07:26:21,568082][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov -->
<!--                     [2026-01-08 07:26:21,568770][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host] -->
<!--                     [2026-01-08 07:26:21,569557][I][ezpz/launch:367:build_executable] Building command to execute by piecing together: -->
<!--                     [2026-01-08 07:26:21,569959][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 -->
<!--                     [2026-01-08 07:26:21,570821][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 demo.py -->
<!--                     [2026-01-08 07:26:21,571548][I][ezpz/launch:433:launch] Took: 2.11 seconds to build command. -->
<!--                     [2026-01-08 07:26:21,571918][I][ezpz/launch:436:launch] Executing: -->
<!--                     mpiexec -->
<!--                     --envall -->
<!--                     --np=24 -->
<!--                     --ppn=12 -->
<!--                     --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov -->
<!--                     --no-vni -->
<!--                     --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 -->
<!--                     python3 -->
<!--                     demo.py -->
<!--                     [2026-01-08 07:26:21,573262][I][ezpz/launch:220:get_aurora_filters] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG -->
<!--                     [2026-01-08 07:26:21,573781][I][ezpz/launch:443:launch] Execution started @ 2026-01-08-072621... -->
<!--                     [2026-01-08 07:26:21,574195][I][ezpz/launch:138:run_command] Caught 24 filters -->
<!--                     [2026-01-08 07:26:21,574532][I][ezpz/launch:139:run_command] Running command: -->
<!--                     mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 demo.py -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131587 rank 12 0: mask 0x1c -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131588 rank 13 1: mask 0x1c00 -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131589 rank 14 2: mask 0x1c0000 -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131590 rank 15 3: mask 0x1c000000 -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131591 rank 16 4: mask 0x1c00000000 -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131592 rank 17 5: mask 0x1c0000000000 -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131593 rank 18 6: mask 0x1c0000000000000 -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131594 rank 19 7: mask 0x1c000000000000000 -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131595 rank 20 8: mask 0x1c00000000000000000 -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131596 rank 21 9: mask 0x1c0000000000000000000 -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131597 rank 22 10: mask 0x1c000000000000000000000 -->
<!--                     cpubind:list x4604c5s3b0n0 pid 131598 rank 23 11: mask 0x1c00000000000000000000000 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121225 rank 0 0: mask 0x1c -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121226 rank 1 1: mask 0x1c00 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121227 rank 2 2: mask 0x1c0000 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121228 rank 3 3: mask 0x1c000000 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121229 rank 4 4: mask 0x1c00000000 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121230 rank 5 5: mask 0x1c0000000000 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121231 rank 6 6: mask 0x1c0000000000000 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121232 rank 7 7: mask 0x1c000000000000000 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121233 rank 8 8: mask 0x1c00000000000000000 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121234 rank 9 9: mask 0x1c0000000000000000000 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121235 rank 10 10: mask 0x1c000000000000000000000 -->
<!--                     cpubind:list x4604c5s2b0n0 pid 121236 rank 11 11: mask 0x1c00000000000000000000000 -->
<!--                     Using [24 / 24] available "xpu" devices !! -->
<!--                     Hello from rank 0 / 24 on xpu! -->
<!--                     [2026-01-08 07:26:33,060432][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2026-01-08-072633]---- -->
<!--                     [2026-01-08 07:26:33,061512][I][ezpz/launch:448:launch] Execution finished with 0. -->
<!--                     [2026-01-08 07:26:33,062045][I][ezpz/launch:449:launch] Executing finished in 11.49 seconds. -->
<!--                     [2026-01-08 07:26:33,062531][I][ezpz/launch:450:launch] Took 11.49 seconds to run. Exiting. -->
<!--                     took: 22s -->
<!--                     ``` -->
