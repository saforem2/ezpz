# 📝 Ready-to-go Examples

## Train MLP with DDP on MNIST

Train a simple fully connected (`torch.nn.Linear`) network using DDP
on the MNIST dataset.

See: [\[docs\]](../python/Code-Reference/test_dist.md), [\[source\]](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py)
1. 📘 Docs: [test-dist](../python/Code-Reference/test_dist.md)
1. 🐍 Source: [src/ezpz/test_dist.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py)

```bash
# or, equivalently: ezpz test
ezpz launch python3 -m ezpz.test_dist
```

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.test_dist --help
usage: test_dist.py [-h] [--warmup WARMUP] [--tp TP] [--pp PP] [--deepspeed_config DEEPSPEED_CONFIG] [--cp CP] [--backend BACKEND]
                    [--pyinstrument-profiler] [-p] [--rank-zero-only] [--pytorch-profiler-wait PYTORCH_PROFILER_WAIT]
                    [--pytorch-profiler-warmup PYTORCH_PROFILER_WARMUP] [--pytorch-profiler-active PYTORCH_PROFILER_ACTIVE]
                    [--pytorch-profiler-repeat PYTORCH_PROFILER_REPEAT] [--profile-memory] [--record-shapes] [--with-stack]
                    [--with-flops] [--with-modules] [--acc-events] [--train-iters TRAIN_ITERS] [--log-freq LOG_FREQ]
                    [--print-freq PRINT_FREQ] [--batch-size BATCH_SIZE] [--input-size INPUT_SIZE] [--output-size OUTPUT_SIZE]
                    [--layer-sizes LAYER_SIZES] [--dtype DTYPE] [--dataset DATASET] [--dataset-root DATASET_ROOT]
                    [--num-workers NUM_WORKERS] [--no-distributed-history]

ezpz test: A simple PyTorch distributed smoke test Trains a simple MLP on MNIST dataset using DDP. NOTE: `ezpz test` is a lightweight
wrapper around: `ezpz launch python3 -m ezpz.test_dist`

options:
    -h, --help            show this help message and exit
    --warmup WARMUP       Warmup iterations
    --tp TP               Tensor parallel size
    --pp PP               Pipeline length
    --deepspeed_config DEEPSPEED_CONFIG
                        Deepspeed config file
    --cp CP               Context parallel size
    --backend BACKEND     Backend (DDP, DeepSpeed, etc.)
    --pyinstrument-profiler
                        Profile the training loop
    -p, --profile         Use PyTorch profiler
    --rank-zero-only      Run profiler only on rank 0
    --pytorch-profiler-wait PYTORCH_PROFILER_WAIT
                        Wait time before starting the PyTorch profiler
    --pytorch-profiler-warmup PYTORCH_PROFILER_WARMUP
                        Warmup iterations for the PyTorch profiler
    --pytorch-profiler-active PYTORCH_PROFILER_ACTIVE
                        Active iterations for the PyTorch profiler
    --pytorch-profiler-repeat PYTORCH_PROFILER_REPEAT
                        Repeat iterations for the PyTorch profiler
    --profile-memory      Profile memory usage
    --record-shapes       Record shapes in the profiler
    --with-stack          Include stack traces in the profiler
    --with-flops          Include FLOPs in the profiler
    --with-modules        Include module information in the profiler
    --acc-events          Accumulate events in the profiler
    --train-iters TRAIN_ITERS, --train_iters TRAIN_ITERS
                        Number of training iterations
    --log-freq LOG_FREQ, --log_freq LOG_FREQ
                        Logging frequency
    --print-freq PRINT_FREQ, --print_freq PRINT_FREQ
                        Printing frequency
    --batch-size BATCH_SIZE
                        Batch size
    --input-size INPUT_SIZE
                        Input size
    --output-size OUTPUT_SIZE
                        Output size
    --layer-sizes LAYER_SIZES
                        Comma-separated list of layer sizes
    --dtype DTYPE         Data type (fp16, float16, bfloat16, bf16, float32, etc.)
    --dataset DATASET     Dataset to use for training (e.g., mnist).
    --dataset-root DATASET_ROOT
                        Directory to cache dataset downloads.
    --num-workers NUM_WORKERS
                        Number of dataloader workers to use.
    --no-distributed-history
                        Disable distributed history aggregation
```

</details>


## Train CNN with FSDP on MNIST

See:

1. 📘 Docs: [examples/FSDP](../python/Code-Reference/examples/fsdp.md)
1. 🐍 Source: [src/ezpz/examples/fsdp.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py)

```bash
ezpz launch python3 -m ezpz.examples.fsdp
```

<details closed><summary><code>--help</code></summary>

```bash

usage: fsdp.py [-h] [--num-workers N]
            [--dataset {MNIST,OpenImages,ImageNet,ImageNet1k}]
            [--batch-size N] [--dtype D] [--test-batch-size N] [--epochs N]
            [--lr LR] [--gamma M] [--seed S] [--save-model]
            [--data-prefix DATA_PREFIX]

PyTorch MNIST Example using FSDP

options:
-h, --help            show this help message and exit
--num-workers N       number of data loading workers (default: 4)
--dataset {MNIST,OpenImages,ImageNet,ImageNet1k}
                        Dataset to use (default: MNIST)
--batch-size N        input batch size for training (default: 64)
--dtype D             Datatype for training (default=bf16).
--test-batch-size N   input batch size for testing (default: 1000)
--epochs N            number of epochs to train (default: 10)
--lr LR               learning rate (default: 1e-3)
--gamma M             Learning rate step gamma (default: 0.7)
--seed S              random seed (default: 1)
--save-model          For Saving the current Model
--data-prefix DATA_PREFIX
                        data directory prefix

```

</details>

## Train ViT with FSDP on MNIST

See:

1. 📘 Docs: [examples/ViT](../python/Code-Reference/examples/vit.md)
1. 🐍 Source: [src/ezpz/examples/vit.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py)

```bash
ezpz launch python3 -m ezpz.examples.vit --compile # --fsdp
```

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.examples.vit --help
usage: ezpz.examples.vit [-h] [--img_size IMG_SIZE] [--batch_size BATCH_SIZE]
                        [--num_heads NUM_HEADS] [--head_dim HEAD_DIM]
                        [--hidden-dim HIDDEN_DIM] [--mlp-dim MLP_DIM]
                        [--dropout DROPOUT]
                        [--attention-dropout ATTENTION_DROPOUT]
                        [--num_classes NUM_CLASSES] [--dataset {fake,mnist}]
                        [--depth DEPTH] [--patch_size PATCH_SIZE]
                        [--dtype DTYPE] [--compile]
                        [--num_workers NUM_WORKERS] [--max_iters MAX_ITERS]
                        [--warmup WARMUP] [--attn_type {native,sdpa}]
                        [--cuda_sdpa_backend {flash_sdp,mem_efficient_sdp,math_sdp,cudnn_sdp,all}]
                        [--fsdp]

Train a simple ViT

options:
    -h, --help            show this help message and exit
    --img_size IMG_SIZE, --img-size IMG_SIZE
                        Image size
    --batch_size BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
    --num_heads NUM_HEADS, --num-heads NUM_HEADS
                        Number of heads
    --head_dim HEAD_DIM, --head-dim HEAD_DIM
                        Hidden Dimension
    --hidden-dim HIDDEN_DIM, --hidden_dim HIDDEN_DIM
                        Hidden Dimension
    --mlp-dim MLP_DIM, --mlp_dim MLP_DIM
                        MLP Dimension
    --dropout DROPOUT     Dropout rate
    --attention-dropout ATTENTION_DROPOUT, --attention_dropout ATTENTION_DROPOUT
                        Attention Dropout rate
    --num_classes NUM_CLASSES, --num-classes NUM_CLASSES
                        Number of classes
    --dataset {fake,mnist}
                        Dataset to use
    --depth DEPTH         Depth
    --patch_size PATCH_SIZE, --patch-size PATCH_SIZE
                        Patch size
    --dtype DTYPE         Data type
    --compile             Compile model
    --num_workers NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of workers
    --max_iters MAX_ITERS, --max-iters MAX_ITERS
                        Maximum iterations
    --warmup WARMUP       Warmup iterations (or fraction) before starting to
                        collect metrics.
    --attn_type {native,sdpa}, --attn-type {native,sdpa}
                        Attention function to use.
    --cuda_sdpa_backend {flash_sdp,mem_efficient_sdp,math_sdp,cudnn_sdp,all}, --cuda-sdpa-backend {flash_sdp,mem_efficient_sdp,math_sdp,cudnn_sdp,all}
                        CUDA SDPA backend to use.
    --fsdp                Use FSDP

```

</details>


## Train Transformer with FSDP and TP on HF Datasets

FSDP Example with Tensor Parallelism

See:

1. 📘 Docs: [examples/FSDP TP](../python/Code-Reference/examples/fsdp_tp.md)
1. 🐍 Source: [src/ezpz/examples/fsdp.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py)

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp \
    --tp=2 \
    --epochs=5 \
    --batch-size=2 \
    --dataset=eliplutchok/fineweb-small-sample \
```

<details closed><summary><code>--help</code></summary>

```bash
usage: fsdp_tp.py [-h] [--dim DIM] [--n-layers N_LAYERS] [--n-heads N_HEADS]
                [--n-kv-heads N_KV_HEADS] [--multiple-of MULTIPLE_OF]
                [--ffn-dim-multiplier FFN_DIM_MULTIPLIER]
                [--norm-eps NORM_EPS] [--vocab-size VOCAB_SIZE]
                [--seq-length SEQ_LENGTH] [--lr LR] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE]
                [--test-batch-size TEST_BATCH_SIZE]
                [--num-workers NUM_WORKERS] [--seed SEED] [--tp TP]
                [--sharding-strategy SHARDING_STRATEGY]
                [--max-grad-norm MAX_GRAD_NORM] [--outdir OUTDIR]
                [--dataset DATASET] [--tokenizer_name TOKENIZER_NAME]
                [--model_name_or_path MODEL_NAME_OR_PATH]
                [--hf-split HF_SPLIT] [--hf-text-column HF_TEXT_COLUMN]
                [--hf-limit HF_LIMIT] [--seq-len SEQ_LEN]
                [--max-seq-len MAX_SEQ_LEN] [--depth-init DEPTH_INIT]
                [--fp32]

2D Parallel Training

options:
-h, --help            show this help message and exit
--dim DIM
--n-layers N_LAYERS
--n-heads N_HEADS
--n-kv-heads N_KV_HEADS
--multiple-of MULTIPLE_OF
--ffn-dim-multiplier FFN_DIM_MULTIPLIER
--norm-eps NORM_EPS
--vocab-size VOCAB_SIZE
--seq-length SEQ_LENGTH
--lr LR
--epochs EPOCHS
--batch-size BATCH_SIZE
--test-batch-size TEST_BATCH_SIZE
--num-workers NUM_WORKERS
--seed SEED
--tp TP
--sharding-strategy SHARDING_STRATEGY
--max-grad-norm MAX_GRAD_NORM
--outdir OUTDIR
--dataset DATASET
--tokenizer_name TOKENIZER_NAME
--model_name_or_path MODEL_NAME_OR_PATH
--hf-split HF_SPLIT, --hf_split HF_SPLIT
                        Dataset split to load.
--hf-text-column HF_TEXT_COLUMN, --hf_text_column HF_TEXT_COLUMN
                        Column containing raw text in the dataset.
--hf-limit HF_LIMIT, --hf_limit HF_LIMIT
                        Number of rows to sample from the HF dataset for
                        quick experiments.
--seq-len SEQ_LEN
--max-seq-len MAX_SEQ_LEN
--depth-init DEPTH_INIT
--fp32                Disable mixed precision (use fp32) for debugging NaNs.
```

</details>

## Train Diffusion LLM with FSDP on HF Datasets

See:

1. 📘 Docs: [examples/Diffusion](../python/Code-Reference/examples/diffusion.md)
1. 🐍 Source: [src/ezpz/examples/diffusion.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py)

```bash
ezpz launch python3 -m ezpz.examples.diffusion --batch_size 1 --hf_dataset stanfordnlp/imdb
```

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.examples.diffusion --help
usage: diffusion.py [-h] [--batch-size BATCH_SIZE] [--dtype DTYPE]
                    [--extra-text [EXTRA_TEXT ...]] [--fsdp]
                    [--fsdp-mixed-precision] [--hidden HIDDEN]
                    [--hf-dataset HF_DATASET] [--hf-split HF_SPLIT]
                    [--hf-text-column HF_TEXT_COLUMN] [--hf-limit HF_LIMIT]
                    [--log_freq LOG_FREQ] [--outdir OUTDIR]
                    [--samples SAMPLES] [--seed SEED] [--seq-len SEQ_LEN]
                    [--timesteps TIMESTEPS] [--train-steps TRAIN_STEPS]
                    [--lr LR]

Tiny diffusion example for text generation.

options:
-h, --help            show this help message and exit
--batch-size BATCH_SIZE
--dtype DTYPE
--extra-text [EXTRA_TEXT ...]
                        Additional sentences to add to the tiny corpus.
--fsdp                Enable FSDP wrapping (requires WORLD_SIZE>1 and
                        torch.distributed init).
--fsdp-mixed-precision
                        Use bfloat16 parameters with FSDP for speed (defaults
                        to float32).
--hidden HIDDEN
--hf-dataset HF_DATASET
                        Optional Hugging Face dataset name (e.g., 'ag_news').
                        When set, replaces the toy corpus.
--hf-split HF_SPLIT   Dataset split to load.
--hf-text-column HF_TEXT_COLUMN
                        Column containing raw text in the dataset.
--hf-limit HF_LIMIT   Number of rows to sample from the HF dataset for quick
                        experiments.
--log_freq LOG_FREQ
--outdir OUTDIR
--samples SAMPLES
--seed SEED
--seq-len SEQ_LEN
--timesteps TIMESTEPS
--train-steps TRAIN_STEPS
--lr LR
```

</details>

## Train or Fine-Tune an LLM with FSDP and HF Trainer on HF Datasets

See:

1. 📘 Docs: [examples/HF Trainer](../python/Code-Reference/examples/hf_trainer.md)
    - [Comparison between Aurora/Polaris at ALCF](../notes/hf-trainer-comparison.md)
1. 🐍 Source: [src/ezpz/examples/hf_trainer.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py)

```bash
ezpz launch python3 -m ezpz.examples.hf_trainer \
    --streaming \
    --dataset_name=eliplutchok/fineweb-small-sample \
    --tokenizer_name meta-llama/Llama-3.2-1B \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --bf16=true \
    --do_train=true \
    --do_eval=true \
    --report-to=wandb \
    --logging-steps=1 \
    --include-tokens-per-second=true \
    --max-steps=50000 \
    --include-num-input-tokens-seen=true \
    --optim=adamw_torch \
    --logging-first-step \
    --include-for-metrics='inputs,loss' \
    --max-eval-samples=50 \
    --per_device_train_batch_size=1 \
    --block-size=8192 \
    --gradient_checkpointing=true \
    --fsdp=shard_grad_op
```

<details closed><summary><code>--help</code></summary>

```bash
usage: hf_trainer.py [-h] [--wandb_project_name WANDB_PROJECT_NAME]
                    [--model_name_or_path MODEL_NAME_OR_PATH]
                    [--model_type MODEL_TYPE]
                    [--config_overrides CONFIG_OVERRIDES]
                    [--config_name CONFIG_NAME]
                    [--tokenizer_name TOKENIZER_NAME] [--cache_dir CACHE_DIR]
                    [--use_fast_tokenizer [USE_FAST_TOKENIZER]]
                    [--no_use_fast_tokenizer]
                    [--model_revision MODEL_REVISION] [--token TOKEN]
                    [--trust_remote_code [TRUST_REMOTE_CODE]]
                    [--torch_dtype {auto,bfloat16,float16,float32}]
                    [--low_cpu_mem_usage [LOW_CPU_MEM_USAGE]]
                    [--data_path DATA_PATH] [--dataset_name DATASET_NAME]
                    [--dataset_config_name DATASET_CONFIG_NAME]
                    [--train_split_str TRAIN_SPLIT_STR]
                    [--train_split_name TRAIN_SPLIT_NAME]
                    [--validation_split_name VALIDATION_SPLIT_NAME]
                    [--validation_split_str VALIDATION_SPLIT_STR]
                    [--test_split_name TEST_SPLIT_NAME]
                    [--test_split_str TEST_SPLIT_STR]
                    [--train_file TRAIN_FILE]
                    [--validation_file VALIDATION_FILE]
                    [--max_train_samples MAX_TRAIN_SAMPLES]
                    [--max_eval_samples MAX_EVAL_SAMPLES]
                    [--streaming [STREAMING]] [--block_size BLOCK_SIZE]
                    [--overwrite_cache [OVERWRITE_CACHE]]
                    [--validation_split_percentage VALIDATION_SPLIT_PERCENTAGE]
                    [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS]
                    [--keep_linebreaks [KEEP_LINEBREAKS]]
                    [--no_keep_linebreaks] [--output_dir OUTPUT_DIR]
                    [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]]
                    [--do_train [DO_TRAIN]] [--do_eval [DO_EVAL]]
                    [--do_predict [DO_PREDICT]]
                    [--eval_strategy {no,steps,epoch}]
                    [--prediction_loss_only [PREDICTION_LOSS_ONLY]]
                    [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
                    [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
                    [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                    [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                    [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                    [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS]
                    [--eval_delay EVAL_DELAY]
                    [--torch_empty_cache_steps TORCH_EMPTY_CACHE_STEPS]
                    [--learning_rate LEARNING_RATE]
                    [--weight_decay WEIGHT_DECAY] [--adam_beta1 ADAM_BETA1]
                    [--adam_beta2 ADAM_BETA2] [--adam_epsilon ADAM_EPSILON]
                    [--max_grad_norm MAX_GRAD_NORM]
                    [--num_train_epochs NUM_TRAIN_EPOCHS]
                    [--max_steps MAX_STEPS]
                    [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,cosine_warmup_with_min_lr,warmup_stable_decay}]
                    [--lr_scheduler_kwargs LR_SCHEDULER_KWARGS]
                    [--warmup_ratio WARMUP_RATIO]
                    [--warmup_steps WARMUP_STEPS]
                    [--log_level {detail,debug,info,warning,error,critical,passive}]
                    [--log_level_replica {detail,debug,info,warning,error,critical,passive}]
                    [--log_on_each_node [LOG_ON_EACH_NODE]]
                    [--no_log_on_each_node] [--logging_dir LOGGING_DIR]
                    [--logging_strategy {no,steps,epoch}]
                    [--logging_first_step [LOGGING_FIRST_STEP]]
                    [--logging_steps LOGGING_STEPS]
                    [--logging_nan_inf_filter [LOGGING_NAN_INF_FILTER]]
                    [--no_logging_nan_inf_filter]
                    [--save_strategy {no,steps,epoch,best}]
                    [--save_steps SAVE_STEPS]
                    [--save_total_limit SAVE_TOTAL_LIMIT]
                    [--save_safetensors [SAVE_SAFETENSORS]]
                    [--no_save_safetensors]
                    [--save_on_each_node [SAVE_ON_EACH_NODE]]
                    [--save_only_model [SAVE_ONLY_MODEL]]
                    [--restore_callback_states_from_checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT]]
                    [--no_cuda [NO_CUDA]] [--use_cpu [USE_CPU]]
                    [--use_mps_device [USE_MPS_DEVICE]] [--seed SEED]
                    [--data_seed DATA_SEED] [--jit_mode_eval [JIT_MODE_EVAL]]
                    [--bf16 [BF16]] [--fp16 [FP16]]
                    [--fp16_opt_level FP16_OPT_LEVEL]
                    [--half_precision_backend {auto,apex,cpu_amp}]
                    [--bf16_full_eval [BF16_FULL_EVAL]]
                    [--fp16_full_eval [FP16_FULL_EVAL]] [--tf32 TF32]
                    [--local_rank LOCAL_RANK]
                    [--ddp_backend {nccl,gloo,mpi,ccl,hccl,cncl,mccl}]
                    [--tpu_num_cores TPU_NUM_CORES]
                    [--tpu_metrics_debug [TPU_METRICS_DEBUG]]
                    [--debug DEBUG [DEBUG ...]]
                    [--dataloader_drop_last [DATALOADER_DROP_LAST]]
                    [--eval_steps EVAL_STEPS]
                    [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                    [--dataloader_prefetch_factor DATALOADER_PREFETCH_FACTOR]
                    [--past_index PAST_INDEX] [--run_name RUN_NAME]
                    [--disable_tqdm DISABLE_TQDM]
                    [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]]
                    [--no_remove_unused_columns]
                    [--label_names LABEL_NAMES [LABEL_NAMES ...]]
                    [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]]
                    [--metric_for_best_model METRIC_FOR_BEST_MODEL]
                    [--greater_is_better GREATER_IS_BETTER]
                    [--ignore_data_skip [IGNORE_DATA_SKIP]] [--fsdp FSDP]
                    [--fsdp_min_num_params FSDP_MIN_NUM_PARAMS]
                    [--fsdp_config FSDP_CONFIG]
                    [--fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP]
                    [--accelerator_config ACCELERATOR_CONFIG]
                    [--parallelism_config PARALLELISM_CONFIG]
                    [--deepspeed DEEPSPEED]
                    [--label_smoothing_factor LABEL_SMOOTHING_FACTOR]
                    [--optim {adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_torch_npu_fused,adamw_apex_fused,adafactor,adamw_anyprecision,adamw_torch_4bit,adamw_torch_8bit,ademamix,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,ademamix_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_ademamix_32bit,paged_ademamix_8bit,paged_lion_32bit,paged_lion_8bit,rmsprop,rmsprop_bnb,rmsprop_bnb_8bit,rmsprop_bnb_32bit,galore_adamw,galore_adamw_8bit,galore_adafactor,galore_adamw_layerwise,galore_adamw_8bit_layerwise,galore_adafactor_layerwise,lomo,adalomo,grokadamw,schedule_free_radam,schedule_free_adamw,schedule_free_sgd,apollo_adamw,apollo_adamw_layerwise,stable_adamw}]
                    [--optim_args OPTIM_ARGS] [--adafactor [ADAFACTOR]]
                    [--group_by_length [GROUP_BY_LENGTH]]
                    [--length_column_name LENGTH_COLUMN_NAME]
                    [--report_to REPORT_TO] [--project PROJECT]
                    [--trackio_space_id TRACKIO_SPACE_ID]
                    [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS]
                    [--ddp_bucket_cap_mb DDP_BUCKET_CAP_MB]
                    [--ddp_broadcast_buffers DDP_BROADCAST_BUFFERS]
                    [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]]
                    [--no_dataloader_pin_memory]
                    [--dataloader_persistent_workers [DATALOADER_PERSISTENT_WORKERS]]
                    [--skip_memory_metrics [SKIP_MEMORY_METRICS]]
                    [--no_skip_memory_metrics]
                    [--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]]
                    [--push_to_hub [PUSH_TO_HUB]]
                    [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                    [--hub_model_id HUB_MODEL_ID]
                    [--hub_strategy {end,every_save,checkpoint,all_checkpoints}]
                    [--hub_token HUB_TOKEN]
                    [--hub_private_repo HUB_PRIVATE_REPO]
                    [--hub_always_push [HUB_ALWAYS_PUSH]]
                    [--hub_revision HUB_REVISION]
                    [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                    [--gradient_checkpointing_kwargs GRADIENT_CHECKPOINTING_KWARGS]
                    [--include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS]]
                    [--include_for_metrics INCLUDE_FOR_METRICS [INCLUDE_FOR_METRICS ...]]
                    [--eval_do_concat_batches [EVAL_DO_CONCAT_BATCHES]]
                    [--no_eval_do_concat_batches]
                    [--fp16_backend {auto,apex,cpu_amp}]
                    [--push_to_hub_model_id PUSH_TO_HUB_MODEL_ID]
                    [--push_to_hub_organization PUSH_TO_HUB_ORGANIZATION]
                    [--push_to_hub_token PUSH_TO_HUB_TOKEN]
                    [--mp_parameters MP_PARAMETERS]
                    [--auto_find_batch_size [AUTO_FIND_BATCH_SIZE]]
                    [--full_determinism [FULL_DETERMINISM]]
                    [--torchdynamo TORCHDYNAMO] [--ray_scope RAY_SCOPE]
                    [--ddp_timeout DDP_TIMEOUT]
                    [--torch_compile [TORCH_COMPILE]]
                    [--torch_compile_backend TORCH_COMPILE_BACKEND]
                    [--torch_compile_mode TORCH_COMPILE_MODE]
                    [--include_tokens_per_second [INCLUDE_TOKENS_PER_SECOND]]
                    [--include_num_input_tokens_seen [INCLUDE_NUM_INPUT_TOKENS_SEEN]]
                    [--neftune_noise_alpha NEFTUNE_NOISE_ALPHA]
                    [--optim_target_modules OPTIM_TARGET_MODULES]
                    [--batch_eval_metrics [BATCH_EVAL_METRICS]]
                    [--eval_on_start [EVAL_ON_START]]
                    [--use_liger_kernel [USE_LIGER_KERNEL]]
                    [--liger_kernel_config LIGER_KERNEL_CONFIG]
                    [--eval_use_gather_object [EVAL_USE_GATHER_OBJECT]]
                    [--average_tokens_across_devices [AVERAGE_TOKENS_ACROSS_DEVICES]]
                    [--no_average_tokens_across_devices]

options:
-h, --help            show this help message and exit
--wandb_project_name WANDB_PROJECT_NAME, --wandb-project-name WANDB_PROJECT_NAME
                        The name of the wandb project to use. If not
                        specified, will use the model name. (default: None)
--model_name_or_path MODEL_NAME_OR_PATH, --model-name-or-path MODEL_NAME_OR_PATH
                        The model checkpoint for weights initialization. Dont
                        set if you want to train a model from scratch.
                        (default: None)
--model_type MODEL_TYPE, --model-type MODEL_TYPE
                        If training from scratch, pass a model type from the
                        list: h, t, t, p, s, :, /, /, h, u, g, g, i, n, g, f,
                        a, c, e, ., c, o, /, d, o, c, s, /, t, r, a, n, s, f,
                        o, r, m, e, r, s, /, e, n, /, m, o, d, e, l, s
                        (default: None)
--config_overrides CONFIG_OVERRIDES, --config-overrides CONFIG_OVERRIDES
                        Override some existing default config settings when a
                        model is trained from scratch. Example: n_embd=10,resi
                        d_pdrop=0.2,scale_attn_weights=false,summary_type=cls_
                        index (default: None)
--config_name CONFIG_NAME, --config-name CONFIG_NAME
                        Pretrained config name or path if not the same as
                        model_name (default: None)
--tokenizer_name TOKENIZER_NAME, --tokenizer-name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as
                        model_name (default: None)
--cache_dir CACHE_DIR, --cache-dir CACHE_DIR
                        Where do you want to store the pretrained models
                        downloaded from huggingface.co (default: None)
--use_fast_tokenizer [USE_FAST_TOKENIZER], --use-fast-tokenizer [USE_FAST_TOKENIZER]
                        Whether to use one of the fast tokenizer (backed by
                        the tokenizers library) or not. (default: True)
--no_use_fast_tokenizer, --no-use-fast-tokenizer
                        Whether to use one of the fast tokenizer (backed by
                        the tokenizers library) or not. (default: False)
--model_revision MODEL_REVISION, --model-revision MODEL_REVISION
                        The specific model version to use (can be a branch
                        name, tag name or commit id). (default: main)
--token TOKEN         The token to use as HTTP bearer authorization for
                        remote files. If not specified, will use the token
                        generated when running `huggingface-cli login` (stored
                        in `~/.huggingface`). (default: None)
--trust_remote_code [TRUST_REMOTE_CODE], --trust-remote-code [TRUST_REMOTE_CODE]
                        Whether to trust the execution of code from
                        datasets/models defined on the Hub. This option should
                        only be set to `True` for repositories you trust and
                        in which you have read the code, as it will execute
                        code present on the Hub on your local machine.
                        (default: False)
--torch_dtype {auto,bfloat16,float16,float32}, --torch-dtype {auto,bfloat16,float16,float32}
                        Override the default `torch.dtype` and load the model
                        under this dtype. If `auto` is passed, the dtype will
                        be automatically derived from the models weights.
                        (default: None)
--low_cpu_mem_usage [LOW_CPU_MEM_USAGE], --low-cpu-mem-usage [LOW_CPU_MEM_USAGE]
                        It is an option to create the model as an empty shell,
                        then only materialize its parameters when the
                        pretrained weights are loaded. set True will benefit
                        LLM loading time and RAM consumption. (default: False)
--data_path DATA_PATH, --data-path DATA_PATH
                        Path to the training data. (default: None)
--dataset_name DATASET_NAME, --dataset-name DATASET_NAME
                        The name of the dataset to use (via the datasets
                        library). (default: None)
--dataset_config_name DATASET_CONFIG_NAME, --dataset-config-name DATASET_CONFIG_NAME
                        The configuration name of the dataset to use (via the
                        datasets library). (default: None)
--train_split_str TRAIN_SPLIT_STR, --train-split-str TRAIN_SPLIT_STR
                        The split string to use for the train split (via the
                        datasets library). (default: None)
--train_split_name TRAIN_SPLIT_NAME, --train-split-name TRAIN_SPLIT_NAME
                        The name of the train split to use (via the datasets
                        library). (default: train)
--validation_split_name VALIDATION_SPLIT_NAME, --validation-split-name VALIDATION_SPLIT_NAME
                        The name of the validation split to use (via the
                        datasets library). (default: validation)
--validation_split_str VALIDATION_SPLIT_STR, --validation-split-str VALIDATION_SPLIT_STR
                        The split string to use for the validation split (via
                        the datasets library). (default: None)
--test_split_name TEST_SPLIT_NAME, --test-split-name TEST_SPLIT_NAME
                        The name of the test split to use (via the datasets
                        library). (default: test)
--test_split_str TEST_SPLIT_STR, --test-split-str TEST_SPLIT_STR
                        The split string to use for the test split (via the
                        datasets library). (default: None)
--train_file TRAIN_FILE, --train-file TRAIN_FILE
                        The input training data file (a text file). (default:
                        None)
--validation_file VALIDATION_FILE, --validation-file VALIDATION_FILE
                        An optional input evaluation data file to evaluate the
                        perplexity on (a text file). (default: None)
--max_train_samples MAX_TRAIN_SAMPLES, --max-train-samples MAX_TRAIN_SAMPLES
                        For debugging purposes or quicker training, truncate
                        the number of training examples to this value if set.
                        (default: None)
--max_eval_samples MAX_EVAL_SAMPLES, --max-eval-samples MAX_EVAL_SAMPLES
                        For debugging purposes or quicker training, truncate
                        the number of evaluation examples to this value if
                        set. (default: None)
--streaming [STREAMING]
                        Enable streaming mode (default: False)
--block_size BLOCK_SIZE, --block-size BLOCK_SIZE
                        Optional input sequence length after tokenization. The
                        training dataset will be truncated in block of this
                        size for training. Default to the model max input
                        length for single sentence inputs (take into account
                        special tokens). (default: None)
--overwrite_cache [OVERWRITE_CACHE], --overwrite-cache [OVERWRITE_CACHE]
                        Overwrite the cached training and evaluation sets
                        (default: False)
--validation_split_percentage VALIDATION_SPLIT_PERCENTAGE, --validation-split-percentage VALIDATION_SPLIT_PERCENTAGE
                        The percentage of the train set used as validation set
                        in case theres no validation split (default: 5)
--preprocessing_num_workers PREPROCESSING_NUM_WORKERS, --preprocessing-num-workers PREPROCESSING_NUM_WORKERS
                        The number of processes to use for the preprocessing.
                        (default: None)
--keep_linebreaks [KEEP_LINEBREAKS], --keep-linebreaks [KEEP_LINEBREAKS]
                        Whether to keep line breaks when using TXT files or
                        not. (default: True)
--no_keep_linebreaks, --no-keep-linebreaks
                        Whether to keep line breaks when using TXT files or
                        not. (default: False)
--output_dir OUTPUT_DIR, --output-dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written. Defaults to
                        'trainer_output' if not provided. (default: None)
--overwrite_output_dir [OVERWRITE_OUTPUT_DIR], --overwrite-output-dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output directory. Use
                        this to continue training if output_dir points to a
                        checkpoint directory. (default: False)
--do_train [DO_TRAIN], --do-train [DO_TRAIN]
                        Whether to run training. (default: False)
--do_eval [DO_EVAL], --do-eval [DO_EVAL]
                        Whether to run eval on the dev set. (default: False)
--do_predict [DO_PREDICT], --do-predict [DO_PREDICT]
                        Whether to run predictions on the test set. (default:
                        False)
--eval_strategy {no,steps,epoch}, --eval-strategy {no,steps,epoch}
                        The evaluation strategy to use. (default: no)
--prediction_loss_only [PREDICTION_LOSS_ONLY], --prediction-loss-only [PREDICTION_LOSS_ONLY]
                        When performing evaluation and predictions, only
                        returns the loss. (default: False)
--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE, --per-device-train-batch-size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per device accelerator core/CPU for
                        training. (default: 8)
--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE, --per-device-eval-batch-size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per device accelerator core/CPU for
                        evaluation. (default: 8)
--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE, --per-gpu-train-batch-size PER_GPU_TRAIN_BATCH_SIZE
                        Deprecated, the use of `--per_device_train_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        training. (default: None)
--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE, --per-gpu-eval-batch-size PER_GPU_EVAL_BATCH_SIZE
                        Deprecated, the use of `--per_device_eval_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        evaluation. (default: None)
--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS, --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass. (default: 1)
--eval_accumulation_steps EVAL_ACCUMULATION_STEPS, --eval-accumulation-steps EVAL_ACCUMULATION_STEPS
                        Number of predictions steps to accumulate before
                        moving the tensors to the CPU. (default: None)
--eval_delay EVAL_DELAY, --eval-delay EVAL_DELAY
                        Number of epochs or steps to wait for before the first
                        evaluation can be performed, depending on the
                        eval_strategy. (default: 0)
--torch_empty_cache_steps TORCH_EMPTY_CACHE_STEPS, --torch-empty-cache-steps TORCH_EMPTY_CACHE_STEPS
                        Number of steps to wait before calling
                        `torch.<device>.empty_cache()`.This can help avoid
                        CUDA out-of-memory errors by lowering peak VRAM usage
                        at a cost of about [10% slower performance](https://git hub.com/huggingface/transformers/issues/31372).
                        If left unset or set to None, cache will not be
                        emptied. (default: None)
--learning_rate LEARNING_RATE, --learning-rate LEARNING_RATE
                        The initial learning rate for AdamW. (default: 5e-05)
--weight_decay WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        Weight decay for AdamW if we apply some. (default:
                        0.0)
--adam_beta1 ADAM_BETA1, --adam-beta1 ADAM_BETA1
                        Beta1 for AdamW optimizer (default: 0.9)
--adam_beta2 ADAM_BETA2, --adam-beta2 ADAM_BETA2
                        Beta2 for AdamW optimizer (default: 0.999)
--adam_epsilon ADAM_EPSILON, --adam-epsilon ADAM_EPSILON
                        Epsilon for AdamW optimizer. (default: 1e-08)
--max_grad_norm MAX_GRAD_NORM, --max-grad-norm MAX_GRAD_NORM
                        Max gradient norm. (default: 1.0)
--num_train_epochs NUM_TRAIN_EPOCHS, --num-train-epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform. (default:
                        3.0)
--max_steps MAX_STEPS, --max-steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs. (default: -1)
--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,cosine_warmup_with_min_lr,warmup_stable_decay}, --lr-scheduler-type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,cosine_warmup_with_min_lr,warmup_stable_decay}
                        The scheduler type to use. (default: linear)
--lr_scheduler_kwargs LR_SCHEDULER_KWARGS, --lr-scheduler-kwargs LR_SCHEDULER_KWARGS
                        Extra parameters for the lr_scheduler such as
                        {'num_cycles': 1} for the cosine with hard restarts.
                        (default: {})
--warmup_ratio WARMUP_RATIO, --warmup-ratio WARMUP_RATIO
                        Linear warmup over warmup_ratio fraction of total
                        steps. (default: 0.0)
--warmup_steps WARMUP_STEPS, --warmup-steps WARMUP_STEPS
                        Linear warmup over warmup_steps. (default: 0)
--log_level {detail,debug,info,warning,error,critical,passive}, --log-level {detail,debug,info,warning,error,critical,passive}
                        Logger log level to use on the main node. Possible
                        choices are the log levels as strings: 'debug',
                        'info', 'warning', 'error' and 'critical', plus a
                        'passive' level which doesnt set anything and lets
                        the application set the level. Defaults to 'passive'.
                        (default: passive)
--log_level_replica {detail,debug,info,warning,error,critical,passive}, --log-level-replica {detail,debug,info,warning,error,critical,passive}
                        Logger log level to use on replica nodes. Same choices
                        and defaults as ``log_level`` (default: warning)
--log_on_each_node [LOG_ON_EACH_NODE], --log-on-each-node [LOG_ON_EACH_NODE]
                        When doing a multinode distributed training, whether
                        to log once per node or just once on the main node.
                        (default: True)
--no_log_on_each_node, --no-log-on-each-node
                        When doing a multinode distributed training, whether
                        to log once per node or just once on the main node.
                        (default: False)
--logging_dir LOGGING_DIR, --logging-dir LOGGING_DIR
                        Tensorboard log dir. (default: None)
--logging_strategy {no,steps,epoch}, --logging-strategy {no,steps,epoch}
                        The logging strategy to use. (default: steps)
--logging_first_step [LOGGING_FIRST_STEP], --logging-first-step [LOGGING_FIRST_STEP]
                        Log the first global_step (default: False)
--logging_steps LOGGING_STEPS, --logging-steps LOGGING_STEPS
                        Log every X updates steps. Should be an integer or a
                        float in range `[0,1)`. If smaller than 1, will be
                        interpreted as ratio of total training steps.
                        (default: 500)
--logging_nan_inf_filter [LOGGING_NAN_INF_FILTER], --logging-nan-inf-filter [LOGGING_NAN_INF_FILTER]
                        Filter nan and inf losses for logging. (default: True)
--no_logging_nan_inf_filter, --no-logging-nan-inf-filter
                        Filter nan and inf losses for logging. (default:
                        False)
--save_strategy {no,steps,epoch,best}, --save-strategy {no,steps,epoch,best}
                        The checkpoint save strategy to use. (default: steps)
--save_steps SAVE_STEPS, --save-steps SAVE_STEPS
                        Save checkpoint every X updates steps. Should be an
                        integer or a float in range `[0,1)`. If smaller than
                        1, will be interpreted as ratio of total training
                        steps. (default: 500)
--save_total_limit SAVE_TOTAL_LIMIT, --save-total-limit SAVE_TOTAL_LIMIT
                        If a value is passed, will limit the total amount of
                        checkpoints. Deletes the older checkpoints in
                        `output_dir`. When `load_best_model_at_end` is
                        enabled, the 'best' checkpoint according to
                        `metric_for_best_model` will always be retained in
                        addition to the most recent ones. For example, for
                        `save_total_limit=5` and
                        `load_best_model_at_end=True`, the four last
                        checkpoints will always be retained alongside the best
                        model. When `save_total_limit=1` and
                        `load_best_model_at_end=True`, it is possible that two
                        checkpoints are saved: the last one and the best one
                        (if they are different). Default is unlimited
                        checkpoints (default: None)
--save_safetensors [SAVE_SAFETENSORS], --save-safetensors [SAVE_SAFETENSORS]
                        Use safetensors saving and loading for state dicts
                        instead of default torch.load and torch.save.
                        (default: True)
--no_save_safetensors, --no-save-safetensors
                        Use safetensors saving and loading for state dicts
                        instead of default torch.load and torch.save.
                        (default: False)
--save_on_each_node [SAVE_ON_EACH_NODE], --save-on-each-node [SAVE_ON_EACH_NODE]
                        When doing multi-node distributed training, whether to
                        save models and checkpoints on each node, or only on
                        the main one (default: False)
--save_only_model [SAVE_ONLY_MODEL], --save-only-model [SAVE_ONLY_MODEL]
                        When checkpointing, whether to only save the model, or
                        also the optimizer, scheduler & rng state.Note that
                        when this is true, you wont be able to resume
                        training from checkpoint.This enables you to save
                        storage by not storing the optimizer, scheduler & rng
                        state.You can only load the model using
                        from_pretrained with this option set to True.
                        (default: False)
--restore_callback_states_from_checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT], --restore-callback-states-from-checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT]
                        Whether to restore the callback states from the
                        checkpoint. If `True`, will override callbacks passed
                        to the `Trainer` if they exist in the checkpoint.
                        (default: False)
--no_cuda [NO_CUDA], --no-cuda [NO_CUDA]
                        This argument is deprecated. It will be removed in
                        version 5.0 of 🤗 Transformers. (default: False)
--use_cpu [USE_CPU], --use-cpu [USE_CPU]
                        Whether or not to use cpu. If left to False, we will
                        use the available torch device/backend
                        (cuda/mps/xpu/hpu etc.) (default: False)
--use_mps_device [USE_MPS_DEVICE], --use-mps-device [USE_MPS_DEVICE]
                        This argument is deprecated. `mps` device will be used
                        if available similar to `cuda` device. It will be
                        removed in version 5.0 of 🤗 Transformers (default:
                        False)
--seed SEED           Random seed that will be set at the beginning of
                        training. (default: 42)
--data_seed DATA_SEED, --data-seed DATA_SEED
                        Random seed to be used with data samplers. (default:
                        None)
--jit_mode_eval [JIT_MODE_EVAL], --jit-mode-eval [JIT_MODE_EVAL]
                        Whether or not to use PyTorch jit trace for inference
                        (default: False)
--bf16 [BF16]         Whether to use bf16 (mixed) precision instead of
                        32-bit. Requires Ampere or higher NVIDIA architecture
                        or using CPU (use_cpu) or Ascend NPU. This is an
                        experimental API and it may change. (default: False)
--fp16 [FP16]         Whether to use fp16 (mixed) precision instead of
                        32-bit (default: False)
--fp16_opt_level FP16_OPT_LEVEL, --fp16-opt-level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3']. See details at
                        https://nvidia.github.io/apex/amp.html (default: O1)
--half_precision_backend {auto,apex,cpu_amp}, --half-precision-backend {auto,apex,cpu_amp}
                        The backend to be used for half precision. (default:
                        auto)
--bf16_full_eval [BF16_FULL_EVAL], --bf16-full-eval [BF16_FULL_EVAL]
                        Whether to use full bfloat16 evaluation instead of
                        32-bit. This is an experimental API and it may change.
                        (default: False)
--fp16_full_eval [FP16_FULL_EVAL], --fp16-full-eval [FP16_FULL_EVAL]
                        Whether to use full float16 evaluation instead of
                        32-bit (default: False)
--tf32 TF32           Whether to enable tf32 mode, available in Ampere and
                        newer GPU architectures. This is an experimental API
                        and it may change. (default: None)
--local_rank LOCAL_RANK, --local-rank LOCAL_RANK
                        For distributed training: local_rank (default: -1)
--ddp_backend {nccl,gloo,mpi,ccl,hccl,cncl,mccl}, --ddp-backend {nccl,gloo,mpi,ccl,hccl,cncl,mccl}
                        The backend to be used for distributed training
                        (default: None)
--tpu_num_cores TPU_NUM_CORES, --tpu-num-cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by
                        launcher script) (default: None)
--tpu_metrics_debug [TPU_METRICS_DEBUG], --tpu-metrics-debug [TPU_METRICS_DEBUG]
                        Deprecated, the use of `--debug tpu_metrics_debug` is
                        preferred. TPU: Whether to print debug metrics
                        (default: False)
--debug DEBUG [DEBUG ...]
                        Whether or not to enable debug mode. Current options:
                        `underflow_overflow` (Detect underflow and overflow in
                        activations and weights), `tpu_metrics_debug` (print
                        debug metrics on TPU). (default: None)
--dataloader_drop_last [DATALOADER_DROP_LAST], --dataloader-drop-last [DATALOADER_DROP_LAST]
                        Drop the last incomplete batch if it is not divisible
                        by the batch size. (default: False)
--eval_steps EVAL_STEPS, --eval-steps EVAL_STEPS
                        Run an evaluation every X steps. Should be an integer
                        or a float in range `[0,1)`. If smaller than 1, will
                        be interpreted as ratio of total training steps.
                        (default: None)
--dataloader_num_workers DATALOADER_NUM_WORKERS, --dataloader-num-workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading
                        (PyTorch only). 0 means that the data will be loaded
                        in the main process. (default: 0)
--dataloader_prefetch_factor DATALOADER_PREFETCH_FACTOR, --dataloader-prefetch-factor DATALOADER_PREFETCH_FACTOR
                        Number of batches loaded in advance by each worker. 2
                        means there will be a total of 2 * num_workers batches
                        prefetched across all workers. (default: None)
--past_index PAST_INDEX, --past-index PAST_INDEX
                        If >=0, uses the corresponding part of the output as
                        the past state for next step. (default: -1)
--run_name RUN_NAME, --run-name RUN_NAME
                        An optional descriptor for the run. Notably used for
                        trackio, wandb, mlflow comet and swanlab logging.
                        (default: None)
--disable_tqdm DISABLE_TQDM, --disable-tqdm DISABLE_TQDM
                        Whether or not to disable the tqdm progress bars.
                        (default: None)
--remove_unused_columns [REMOVE_UNUSED_COLUMNS], --remove-unused-columns [REMOVE_UNUSED_COLUMNS]
                        Remove columns not required by the model when using an
                        nlp.Dataset. (default: True)
--no_remove_unused_columns, --no-remove-unused-columns
                        Remove columns not required by the model when using an
                        nlp.Dataset. (default: False)
--label_names LABEL_NAMES [LABEL_NAMES ...], --label-names LABEL_NAMES [LABEL_NAMES ...]
                        The list of keys in your dictionary of inputs that
                        correspond to the labels. (default: None)
--load_best_model_at_end [LOAD_BEST_MODEL_AT_END], --load-best-model-at-end [LOAD_BEST_MODEL_AT_END]
                        Whether or not to load the best model found during
                        training at the end of training. When this option is
                        enabled, the best checkpoint will always be saved. See
                        `save_total_limit` for more. (default: False)
--metric_for_best_model METRIC_FOR_BEST_MODEL, --metric-for-best-model METRIC_FOR_BEST_MODEL
                        The metric to use to compare two different models.
                        (default: None)
--greater_is_better GREATER_IS_BETTER, --greater-is-better GREATER_IS_BETTER
                        Whether the `metric_for_best_model` should be
                        maximized or not. (default: None)
--ignore_data_skip [IGNORE_DATA_SKIP], --ignore-data-skip [IGNORE_DATA_SKIP]
                        When resuming training, whether or not to skip the
                        first epochs and batches to get to the same training
                        data. (default: False)
--fsdp FSDP           Whether or not to use PyTorch Fully Sharded Data
                        Parallel (FSDP) training (in distributed training
                        only). The base option should be `full_shard`,
                        `shard_grad_op` or `no_shard` and you can add CPU-
                        offload to `full_shard` or `shard_grad_op` like this:
                        full_shard offload` or `shard_grad_op offload`. You
                        can add auto-wrap to `full_shard` or `shard_grad_op`
                        with the same syntax: full_shard auto_wrap` or
                        `shard_grad_op auto_wrap`. (default: None)
--fsdp_min_num_params FSDP_MIN_NUM_PARAMS, --fsdp-min-num-params FSDP_MIN_NUM_PARAMS
                        This parameter is deprecated. FSDPs minimum number of
                        parameters for Default Auto Wrapping. (useful only
                        when `fsdp` field is passed). (default: 0)
--fsdp_config FSDP_CONFIG, --fsdp-config FSDP_CONFIG
                        Config to be used with FSDP (Pytorch Fully Sharded
                        Data Parallel). The value is either a fsdp json config
                        file (e.g., `fsdp_config.json`) or an already loaded
                        json file as `dict`. (default: None)
--fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP, --fsdp-transformer-layer-cls-to-wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP
                        This parameter is deprecated. Transformer layer class
                        name (case-sensitive) to wrap, e.g, `BertLayer`,
                        `GPTJBlock`, `T5Block` .... (useful only when `fsdp`
                        flag is passed). (default: None)
--accelerator_config ACCELERATOR_CONFIG, --accelerator-config ACCELERATOR_CONFIG
                        Config to be used with the internal Accelerator object
                        initialization. The value is either a accelerator json
                        config file (e.g., `accelerator_config.json`) or an
                        already loaded json file as `dict`. (default: None)
--parallelism_config PARALLELISM_CONFIG, --parallelism-config PARALLELISM_CONFIG
                        Parallelism configuration for the training run.
                        Requires Accelerate `1.10.1` (default: None)
--deepspeed DEEPSPEED
                        Enable deepspeed and pass the path to deepspeed json
                        config file (e.g. `ds_config.json`) or an already
                        loaded json file as a dict (default: None)
--label_smoothing_factor LABEL_SMOOTHING_FACTOR, --label-smoothing-factor LABEL_SMOOTHING_FACTOR
                        The label smoothing epsilon to apply (zero means no
                        label smoothing). (default: 0.0)
--optim {adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_torch_npu_fused,adamw_apex_fused,adafactor,adamw_anyprecision,adamw_torch_4bit,adamw_torch_8bit,ademamix,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,ademamix_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_ademamix_32bit,paged_ademamix_8bit,paged_lion_32bit,paged_lion_8bit,rmsprop,rmsprop_bnb,rmsprop_bnb_8bit,rmsprop_bnb_32bit,galore_adamw,galore_adamw_8bit,galore_adafactor,galore_adamw_layerwise,galore_adamw_8bit_layerwise,galore_adafactor_layerwise,lomo,adalomo,grokadamw,schedule_free_radam,schedule_free_adamw,schedule_free_sgd,apollo_adamw,apollo_adamw_layerwise,stable_adamw}
                        The optimizer to use. (default: adamw_torch_fused)
--optim_args OPTIM_ARGS, --optim-args OPTIM_ARGS
                        Optional arguments to supply to optimizer. (default:
                        None)
--adafactor [ADAFACTOR]
                        Whether or not to replace AdamW by Adafactor.
                        (default: False)
--group_by_length [GROUP_BY_LENGTH], --group-by-length [GROUP_BY_LENGTH]
                        Whether or not to group samples of roughly the same
                        length together when batching. (default: False)
--length_column_name LENGTH_COLUMN_NAME, --length-column-name LENGTH_COLUMN_NAME
                        Column name with precomputed lengths to use when
                        grouping by length. (default: length)
--report_to REPORT_TO, --report-to REPORT_TO
                        The list of integrations to report the results and
                        logs to. (default: None)
--project PROJECT     The name of the project to use for logging. Currenly,
                        only used by Trackio. (default: huggingface)
--trackio_space_id TRACKIO_SPACE_ID, --trackio-space-id TRACKIO_SPACE_ID
                        The Hugging Face Space ID to deploy to when using
                        Trackio. Should be a complete Space name like
                        'username/reponame' or 'orgname/reponame', or just
                        'reponame' in which case the Space will be created in
                        the currently-logged-in Hugging Face users namespace.
                        If `None`, will log to a local directory. Note that
                        this Space will be public unless you set
                        `hub_private_repo=True` or your organizations default
                        is to create private Spaces. (default: trackio)
--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS, --ddp-find-unused-parameters DDP_FIND_UNUSED_PARAMETERS
                        When using distributed training, the value of the flag
                        `find_unused_parameters` passed to
                        `DistributedDataParallel`. (default: None)
--ddp_bucket_cap_mb DDP_BUCKET_CAP_MB, --ddp-bucket-cap-mb DDP_BUCKET_CAP_MB
                        When using distributed training, the value of the flag
                        `bucket_cap_mb` passed to `DistributedDataParallel`.
                        (default: None)
--ddp_broadcast_buffers DDP_BROADCAST_BUFFERS, --ddp-broadcast-buffers DDP_BROADCAST_BUFFERS
                        When using distributed training, the value of the flag
                        `broadcast_buffers` passed to
                        `DistributedDataParallel`. (default: None)
--dataloader_pin_memory [DATALOADER_PIN_MEMORY], --dataloader-pin-memory [DATALOADER_PIN_MEMORY]
                        Whether or not to pin memory for DataLoader. (default:
                        True)
--no_dataloader_pin_memory, --no-dataloader-pin-memory
                        Whether or not to pin memory for DataLoader. (default:
                        False)
--dataloader_persistent_workers [DATALOADER_PERSISTENT_WORKERS], --dataloader-persistent-workers [DATALOADER_PERSISTENT_WORKERS]
                        If True, the data loader will not shut down the worker
                        processes after a dataset has been consumed once. This
                        allows to maintain the workers Dataset instances
                        alive. Can potentially speed up training, but will
                        increase RAM usage. (default: False)
--skip_memory_metrics [SKIP_MEMORY_METRICS], --skip-memory-metrics [SKIP_MEMORY_METRICS]
                        Whether or not to skip adding of memory profiler
                        reports to metrics. (default: True)
--no_skip_memory_metrics, --no-skip-memory-metrics
                        Whether or not to skip adding of memory profiler
                        reports to metrics. (default: False)
--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP], --use-legacy-prediction-loop [USE_LEGACY_PREDICTION_LOOP]
                        Whether or not to use the legacy prediction_loop in
                        the Trainer. (default: False)
--push_to_hub [PUSH_TO_HUB], --push-to-hub [PUSH_TO_HUB]
                        Whether or not to upload the trained model to the
                        model hub after training. (default: False)
--resume_from_checkpoint RESUME_FROM_CHECKPOINT, --resume-from-checkpoint RESUME_FROM_CHECKPOINT
                        The path to a folder with a valid checkpoint for your
                        model. (default: None)
--hub_model_id HUB_MODEL_ID, --hub-model-id HUB_MODEL_ID
                        The name of the repository to keep in sync with the
                        local `output_dir`. (default: None)
--hub_strategy {end,every_save,checkpoint,all_checkpoints}, --hub-strategy {end,every_save,checkpoint,all_checkpoints}
                        The hub strategy to use when `--push_to_hub` is
                        activated. (default: every_save)
--hub_token HUB_TOKEN, --hub-token HUB_TOKEN
                        The token to use to push to the Model Hub. (default:
                        None)
--hub_private_repo HUB_PRIVATE_REPO, --hub-private-repo HUB_PRIVATE_REPO
                        Whether to make the repo private. If `None` (default),
                        the repo will be public unless the organizations
                        default is private. This value is ignored if the repo
                        already exists. If reporting to Trackio with
                        deployment to Hugging Face Spaces enabled, the same
                        logic determines whether the Space is private.
                        (default: None)
--hub_always_push [HUB_ALWAYS_PUSH], --hub-always-push [HUB_ALWAYS_PUSH]
                        Unless `True`, the Trainer will skip pushes if the
                        previous one wasnt finished yet. (default: False)
--hub_revision HUB_REVISION, --hub-revision HUB_REVISION
                        The revision to use when pushing to the Hub. Can be a
                        branch name, a tag, or a commit hash. (default: None)
--gradient_checkpointing [GRADIENT_CHECKPOINTING], --gradient-checkpointing [GRADIENT_CHECKPOINTING]
                        If True, use gradient checkpointing to save memory at
                        the expense of slower backward pass. (default: False)
--gradient_checkpointing_kwargs GRADIENT_CHECKPOINTING_KWARGS, --gradient-checkpointing-kwargs GRADIENT_CHECKPOINTING_KWARGS
                        Gradient checkpointing key word arguments such as
                        `use_reentrant`. Will be passed to
                        `torch.utils.checkpoint.checkpoint` through
                        `model.gradient_checkpointing_enable`. (default: None)
--include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS], --include-inputs-for-metrics [INCLUDE_INPUTS_FOR_METRICS]
                        This argument is deprecated and will be removed in
                        version 5 of 🤗 Transformers. Use `include_for_metrics`
                        instead. (default: False)
--include_for_metrics INCLUDE_FOR_METRICS [INCLUDE_FOR_METRICS ...], --include-for-metrics INCLUDE_FOR_METRICS [INCLUDE_FOR_METRICS ...]
                        List of strings to specify additional data to include
                        in the `compute_metrics` function.Options: 'inputs',
                        'loss'. (default: [])
--eval_do_concat_batches [EVAL_DO_CONCAT_BATCHES], --eval-do-concat-batches [EVAL_DO_CONCAT_BATCHES]
                        Whether to recursively concat
                        inputs/losses/labels/predictions across batches. If
                        `False`, will instead store them as lists, with each
                        batch kept separate. (default: True)
--no_eval_do_concat_batches, --no-eval-do-concat-batches
                        Whether to recursively concat
                        inputs/losses/labels/predictions across batches. If
                        `False`, will instead store them as lists, with each
                        batch kept separate. (default: False)
--fp16_backend {auto,apex,cpu_amp}, --fp16-backend {auto,apex,cpu_amp}
                        Deprecated. Use half_precision_backend instead
                        (default: auto)
--push_to_hub_model_id PUSH_TO_HUB_MODEL_ID, --push-to-hub-model-id PUSH_TO_HUB_MODEL_ID
                        The name of the repository to which push the
                        `Trainer`. (default: None)
--push_to_hub_organization PUSH_TO_HUB_ORGANIZATION, --push-to-hub-organization PUSH_TO_HUB_ORGANIZATION
                        The name of the organization in with to which push the
                        `Trainer`. (default: None)
--push_to_hub_token PUSH_TO_HUB_TOKEN, --push-to-hub-token PUSH_TO_HUB_TOKEN
                        The token to use to push to the Model Hub. (default:
                        None)
--mp_parameters MP_PARAMETERS, --mp-parameters MP_PARAMETERS
                        Used by the SageMaker launcher to send mp-specific
                        args. Ignored in Trainer (default: )
--auto_find_batch_size [AUTO_FIND_BATCH_SIZE], --auto-find-batch-size [AUTO_FIND_BATCH_SIZE]
                        Whether to automatically decrease the batch size in
                        half and rerun the training loop again each time a
                        CUDA Out-of-Memory was reached (default: False)
--full_determinism [FULL_DETERMINISM], --full-determinism [FULL_DETERMINISM]
                        Whether to call enable_full_determinism instead of
                        set_seed for reproducibility in distributed training.
                        Important: this will negatively impact the
                        performance, so only use it for debugging. (default:
                        False)
--torchdynamo TORCHDYNAMO
                        This argument is deprecated, use
                        `--torch_compile_backend` instead. (default: None)
--ray_scope RAY_SCOPE, --ray-scope RAY_SCOPE
                        The scope to use when doing hyperparameter search with
                        Ray. By default, `"last"` will be used. Ray will then
                        use the last checkpoint of all trials, compare those,
                        and select the best one. However, other options are
                        also available. See the Ray documentation (https://doc
                        s.ray.io/en/latest/tune/api_docs/analysis.html#ray.tun
                        e.ExperimentAnalysis.get_best_trial) for more options.
                        (default: last)
--ddp_timeout DDP_TIMEOUT, --ddp-timeout DDP_TIMEOUT
                        Overrides the default timeout for distributed training
                        (value should be given in seconds). (default: 1800)
--torch_compile [TORCH_COMPILE], --torch-compile [TORCH_COMPILE]
                        If set to `True`, the model will be wrapped in
                        `torch.compile`. (default: False)
--torch_compile_backend TORCH_COMPILE_BACKEND, --torch-compile-backend TORCH_COMPILE_BACKEND
                        Which backend to use with `torch.compile`, passing one
                        will trigger a model compilation. (default: None)
--torch_compile_mode TORCH_COMPILE_MODE, --torch-compile-mode TORCH_COMPILE_MODE
                        Which mode to use with `torch.compile`, passing one
                        will trigger a model compilation. (default: None)
--include_tokens_per_second [INCLUDE_TOKENS_PER_SECOND], --include-tokens-per-second [INCLUDE_TOKENS_PER_SECOND]
                        If set to `True`, the speed metrics will include `tgs`
                        (tokens per second per device). (default: False)
--include_num_input_tokens_seen [INCLUDE_NUM_INPUT_TOKENS_SEEN], --include-num-input-tokens-seen [INCLUDE_NUM_INPUT_TOKENS_SEEN]
                        Whether to track the number of input tokens seen. Can
                        be `'all'` to count all tokens, `'non_padding'` to
                        count only non-padding tokens, or a boolean (`True`
                        maps to `'all'`, `False` to `'no'`). (default: False)
--neftune_noise_alpha NEFTUNE_NOISE_ALPHA, --neftune-noise-alpha NEFTUNE_NOISE_ALPHA
                        Activates neftune noise embeddings into the model.
                        NEFTune has been proven to drastically improve model
                        performances for instruction fine-tuning. Check out
                        the original paper here:
                        https://huggingface.co/papers/2310.05914 and the
                        original code here:
                        https://github.com/neelsjain/NEFTune. Only supported
                        for `PreTrainedModel` and `PeftModel` classes.
                        (default: None)
--optim_target_modules OPTIM_TARGET_MODULES, --optim-target-modules OPTIM_TARGET_MODULES
                        Target modules for the optimizer defined in the
                        `optim` argument. Only used for the GaLore optimizer
                        at the moment. (default: None)
--batch_eval_metrics [BATCH_EVAL_METRICS], --batch-eval-metrics [BATCH_EVAL_METRICS]
                        Break eval metrics calculation into batches to save
                        memory. (default: False)
--eval_on_start [EVAL_ON_START], --eval-on-start [EVAL_ON_START]
                        Whether to run through the entire `evaluation` step at
                        the very beginning of training as a sanity check.
                        (default: False)
--use_liger_kernel [USE_LIGER_KERNEL], --use-liger-kernel [USE_LIGER_KERNEL]
                        Whether or not to enable the Liger Kernel for model
                        training. (default: False)
--liger_kernel_config LIGER_KERNEL_CONFIG, --liger-kernel-config LIGER_KERNEL_CONFIG
                        Configuration to be used for Liger Kernel. When
                        use_liger_kernel=True, this dict is passed as keyword
                        arguments to the `_apply_liger_kernel_to_instance`
                        function, which specifies which kernels to apply.
                        Available options vary by model but typically include:
                        'rope', 'swiglu', 'cross_entropy',
                        'fused_linear_cross_entropy', 'rms_norm', etc. If
                        None, use the default kernel configurations. (default:
                        None)
--eval_use_gather_object [EVAL_USE_GATHER_OBJECT], --eval-use-gather-object [EVAL_USE_GATHER_OBJECT]
                        Whether to run recursively gather object in a nested
                        list/tuple/dictionary of objects from all devices.
                        (default: False)
--average_tokens_across_devices [AVERAGE_TOKENS_ACROSS_DEVICES], --average-tokens-across-devices [AVERAGE_TOKENS_ACROSS_DEVICES]
                        Whether or not to average tokens across devices. If
                        enabled, will use all_reduce to synchronize
                        num_tokens_in_batch for precise loss calculation.
                        Reference: https://github.com/huggingface/transformers
                        /issues/34242 (default: True)
--no_average_tokens_across_devices, --no-average-tokens-across-devices
                        Whether or not to average tokens across devices. If
                        enabled, will use all_reduce to synchronize
                        num_tokens_in_batch for precise loss calculation.
                        Reference: https://github.com/huggingface/transformers
                        /issues/34242 (default: False)
```

</details>
