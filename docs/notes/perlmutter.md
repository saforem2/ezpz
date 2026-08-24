# 🍋 `ezpz` on Perlmutter @ NERSC

!!! tip "Two things worth knowing before you start"

    `module load pytorch` resolves to **2.8.0**, where `--compile`
    aborts every rank ([#236](https://github.com/saforem2/ezpz/issues/236));
    `pytorch/2.13.0` fixes it but needs
    `export MPICH_GPU_SUPPORT_ENABLED=0` or every rank segfaults.
    Both are covered, with benchmark numbers and the SLURM failover
    caveat, in [Perlmutter validation](../guides/perlmutter.md).

1. Submit interactive job on Perlmutter:

    ```bash
    ; NODES=2 ; HRS=02 ; QUEUE=interactive ; salloc --nodes $NODES --qos $QUEUE --time $HRS:30:00 -C 'gpu' --gpus=$(( 4 * NODES )) -A amsc013_g
    ```

1. Load modules:

    ```bash
    module load cudatoolkit/12.9 nccl/2.24.3 pytorch cray-mpich
    ```

1. Navigate to `$SCRATCH` and set environment variables:

    ```bash
    cd $SCRATCH
    export UV_CACHE_DIR="$SCRATCH/.cache/uv"
    export HF_HOME="$SCRATCH/.cache/hf"
    ```


1. Create and activate virtual environment:

    ```bash
    uv venv --python=$(which python3) --system-site-packages
    source .venv/bin/activate
    ```

1. Install `ezpz` (+ `mpi4py`):

    ```bash
    uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz[mpi]"
    ```

1. Run tests:

    ```bash
    # Train MLP on MNIST
    ezpz launch python3 -m ezpz.examples.test

    # Fine Tune LLM
    ezpz launch python3 -m ezpz.examples.hf \
        --dataset_name=eliplutchok/fineweb-small-sample \
        --streaming \
        --model_name_or_path meta-llama/Llama-3.2-1B \
        --bf16=true \
        --do_train=true \
        --do_eval=true \
        --report-to=wandb \
        --logging-steps=1 \
        --include-tokens-per-second=true \
        --max-steps=100 \
        --include-num-input-tokens-seen=true \
        --optim=adamw_torch \
        --logging-first-step \
        --include-for-metrics='inputs,loss' \
        --max-eval-samples=100 \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --block_size=8192 \
        --gradient_checkpointing=true \
        --fsdp=auto_wrap \
        --output_dir=outputs/ezpz.hf_trainer/$(tstamp)
    ```
