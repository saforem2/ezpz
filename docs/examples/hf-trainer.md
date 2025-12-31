# Train or Fine-Tune an LLM with FSDP and HF Trainer on HF Datasets

See:

- 📘 [examples/HF Trainer](../python/Code-Reference/examples/hf_trainer.md)
    - [Comparison between Aurora/Polaris at ALCF](../notes/hf-trainer-comparison.md)
- 🐍 [src/ezpz/examples/hf_trainer.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py)


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

## Help

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

## Output

<details closed><summary>Output on Sunspot</summary>

```bash
#[aurora_frameworks-2025.2.0](ezpz-aurora_frameworks-2025.2.0)
#[/t/d/f/p/s/ezpz][dev][!?] [󰔛  3m18s]
#[12/31/25 @ 13:11:04][x1921c1s5b0n0]
; ezpz launch python3 -m ezpz.examples.hf_trainer --streaming --dataset_name=eliplutchok/fineweb-small-sample --model_name_or_path meta-llama/Llama-3.2-1B --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --include-tokens-per-second=true --max-steps=100 --include-num-input-tokens-seen=true --optim=adamw_torch --logging-first-step --include-for-metrics='inputs,loss' --max-eval-samples=100 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=8192 --gradient_checkpointing=true --fsdp=auto_wrap --output_dir=outputs/ezpz.hf_trainer/$(tstamp)


[2025-12-31 13:11:10,316555][I][numexpr/utils:148:_init_num_threads] Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
[2025-12-31 13:11:10,318860][I][numexpr/utils:151:_init_num_threads] Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
[2025-12-31 13:11:10,319330][I][numexpr/utils:164:_init_num_threads] NumExpr defaulting to 16 threads.
[2025-12-31 13:11:10,544612][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2025-12-31-131110]----
[2025-12-31 13:11:11,591145][I][ezpz/launch:416:launch] Job ID: 12458340
[2025-12-31 13:11:11,591935][I][ezpz/launch:417:launch] nodelist: ['x1921c1s5b0n0', 'x1921c1s7b0n0']
[2025-12-31 13:11:11,592339][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/12458340.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
[2025-12-31 13:11:11,593030][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
[2025-12-31 13:11:11,593768][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
[2025-12-31 13:11:11,594177][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458340.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
[2025-12-31 13:11:11,594907][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 -m ezpz.examples.hf_trainer --streaming --dataset_name=eliplutchok/fineweb-small-sample --model_name_or_path meta-llama/Llama-3.2-1B --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --include-tokens-per-second=true --max-steps=100 --include-num-input-tokens-seen=true --optim=adamw_torch --logging-first-step --include-for-m
etrics=inputs,loss --max-eval-samples=100 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=8192 --gradient_checkpointing=true --fsdp=auto_wrap --output_dir=outputs/ezpz.hf_trainer/2025-12-31-131106
[2025-12-31 13:11:11,596750][I][ezpz/launch:433:launch] Took: 1.54 seconds to build command.
[2025-12-31 13:11:11,597152][I][ezpz/launch:436:launch] Executing:
mpiexec
  --envall
  --np=24
  --ppn=12
  --hostfile=/var/spool/pbs/aux/12458340.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
  --no-vni
  --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
  python3
  -m
  ezpz.examples.hf_trainer
  --streaming
  --dataset_name=eliplutchok/fineweb-small-sample
  --model_name_or_path
  meta-llama/Llama-3.2-1B
  --bf16=true
  --do_train=true
  --do_eval=true
  --report-to=wandb
  --logging-steps=1
  --include-tokens-per-second=true
  --max-steps=100
  --include-num-input-tokens-seen=true
  --optim=adamw_torch
  --logging-first-step
  --include-for-metrics=inputs,loss
  --max-eval-samples=100
  --per_device_train_batch_size=1
  --per_device_eval_batch_size=1
  --block_size=8192
  --gradient_checkpointing=true
  --fsdp=auto_wrap
  --output_dir=outputs/ezpz.hf_trainer/2025-12-31-131106
[2025-12-31 13:11:11,599586][I][ezpz/launch:443:launch] Execution started @ 2025-12-31-131111...
[2025-12-31 13:11:11,600073][I][ezpz/launch:139:run_command] Running command:
 mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458340.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 -m ezpz.examples.hf_trainer --streaming --dataset_name=eliplutchok/fineweb-small-sample --model_name_or_path meta-llama/Llama-3.2-1B --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --inc
lude-tokens-per-second=true --max-steps=100 --include-num-input-tokens-seen=true --optim=adamw_torch --logging-first-step --include-for-metrics=inputs,loss --max-eval-samples=100 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=8192 --gradient_checkpointing=true --fsdp=auto_wrap --output_dir=outputs/ezpz.hf_trainer/2025-12-31-131106
cpubind:list x1921c1s7b0n0 pid 93007 rank 12 0: mask 0x1c
cpubind:list x1921c1s7b0n0 pid 93008 rank 13 1: mask 0x1c00
cpubind:list x1921c1s7b0n0 pid 93009 rank 14 2: mask 0x1c0000
cpubind:list x1921c1s7b0n0 pid 93010 rank 15 3: mask 0x1c000000
cpubind:list x1921c1s7b0n0 pid 93011 rank 16 4: mask 0x1c00000000
cpubind:list x1921c1s7b0n0 pid 93012 rank 17 5: mask 0x1c0000000000
cpubind:list x1921c1s7b0n0 pid 93013 rank 18 6: mask 0x1c0000000000000
cpubind:list x1921c1s7b0n0 pid 93014 rank 19 7: mask 0x1c000000000000000
cpubind:list x1921c1s7b0n0 pid 93015 rank 20 8: mask 0x1c00000000000000000
cpubind:list x1921c1s7b0n0 pid 93016 rank 21 9: mask 0x1c0000000000000000000
cpubind:list x1921c1s7b0n0 pid 93017 rank 22 10: mask 0x1c000000000000000000000
cpubind:list x1921c1s7b0n0 pid 93018 rank 23 11: mask 0x1c00000000000000000000000
cpubind:list x1921c1s5b0n0 pid 87023 rank 0 0: mask 0x1c
cpubind:list x1921c1s5b0n0 pid 87024 rank 1 1: mask 0x1c00
cpubind:list x1921c1s5b0n0 pid 87025 rank 2 2: mask 0x1c0000
cpubind:list x1921c1s5b0n0 pid 87026 rank 3 3: mask 0x1c000000
cpubind:list x1921c1s5b0n0 pid 87027 rank 4 4: mask 0x1c00000000
cpubind:list x1921c1s5b0n0 pid 87028 rank 5 5: mask 0x1c0000000000
cpubind:list x1921c1s5b0n0 pid 87029 rank 6 6: mask 0x1c0000000000000
cpubind:list x1921c1s5b0n0 pid 87030 rank 7 7: mask 0x1c000000000000000
cpubind:list x1921c1s5b0n0 pid 87031 rank 8 8: mask 0x1c00000000000000000
cpubind:list x1921c1s5b0n0 pid 87032 rank 9 9: mask 0x1c0000000000000000000
cpubind:list x1921c1s5b0n0 pid 87033 rank 10 10: mask 0x1c000000000000000000000
cpubind:list x1921c1s5b0n0 pid 87034 rank 11 11: mask 0x1c00000000000000000000000
[2025-12-31 13:11:27,799167][I][ezpz/dist:1501:setup_torch_distributed] Using device=xpu with backend=xccl
[2025-12-31 13:11:27,802253][I][ezpz/dist:1366:setup_torch_DDP] Caught MASTER_PORT=54045 from environment!
[2025-12-31 13:11:27,803043][I][ezpz/dist:1382:setup_torch_DDP] Using torch.distributed.init_process_group with
- master_addr='x1921c1s5b0n0'
- master_port='54045'
- world_size=24
- rank=0
- local_rank=0
- timeout=datetime.timedelta(seconds=3600)
- backend='xccl'
[2025-12-31 13:11:27,804069][I][ezpz/dist:1014:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=24 backend=xccl
[2025-12-31 13:11:28,454410][I][ezpz/dist:1727:setup_torch] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
[2025-12-31 13:11:28,455234][W][ezpz/dist:544:print_dist_setup] Using [24 / 24] available "xpu" devices !!
[2025-12-31 13:11:28,455704][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=0/1][rank=00/23][local_rank=00/11]
[2025-12-31 13:11:28,454824][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=1/1][rank=01/23][local_rank=01/11]
[2025-12-31 13:11:28,454876][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=0/1][rank=02/23][local_rank=02/11]
[2025-12-31 13:11:28,454899][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=1/1][rank=03/23][local_rank=03/11]
[2025-12-31 13:11:28,454911][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=0/1][rank=04/23][local_rank=04/11]
[2025-12-31 13:11:28,454896][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=1/1][rank=05/23][local_rank=05/11]
[2025-12-31 13:11:28,454886][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=0/1][rank=06/23][local_rank=06/11]
[2025-12-31 13:11:28,454893][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=1/1][rank=07/23][local_rank=07/11]
[2025-12-31 13:11:28,454866][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=0/1][rank=08/23][local_rank=08/11]
[2025-12-31 13:11:28,454886][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=1/1][rank=09/23][local_rank=09/11]
[2025-12-31 13:11:28,454902][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=0/1][rank=10/23][local_rank=10/11]
[2025-12-31 13:11:28,454855][I][ezpz/dist:1774:setup_torch] ['x1921c1s5b0n0'][device='xpu'][node=1/1][rank=11/23][local_rank=11/11]
[2025-12-31 13:11:28,454881][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=1/1][rank=15/23][local_rank=03/11]
[2025-12-31 13:11:28,454875][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=0/1][rank=20/23][local_rank=08/11]
[2025-12-31 13:11:28,454929][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=0/1][rank=12/23][local_rank=00/11]
[2025-12-31 13:11:28,454925][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=1/1][rank=13/23][local_rank=01/11]
[2025-12-31 13:11:28,454931][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=0/1][rank=14/23][local_rank=02/11]
[2025-12-31 13:11:28,454901][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=0/1][rank=16/23][local_rank=04/11]
[2025-12-31 13:11:28,454931][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=1/1][rank=17/23][local_rank=05/11]
[2025-12-31 13:11:28,454943][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=0/1][rank=18/23][local_rank=06/11]
[2025-12-31 13:11:28,454938][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=1/1][rank=19/23][local_rank=07/11]
[2025-12-31 13:11:28,454938][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=1/1][rank=21/23][local_rank=09/11]
[2025-12-31 13:11:28,454901][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=0/1][rank=22/23][local_rank=10/11]
[2025-12-31 13:11:28,454938][I][ezpz/dist:1774:setup_torch] ['x1921c1s7b0n0'][device='xpu'][node=1/1][rank=23/23][local_rank=11/11]
[rank18]:[W1231 13:11:28.633965523 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/ipex_2.8.10_xpu_rel_08_18_2025/intel-extension-for-pytorch/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
[2025-12-31 13:11:29,171] [INFO] [real_accelerator.py:260:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2025-12-31 13:11:29,204358][I][_distutils/spawn:77:spawn] icx -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/include -Wformat -Wformat-security -fstack-protector-all -D_FORTIFY_SOURCE=2 -fpic -fPIC -O2 -Wl,-z,noexecstack,-z,relro,-z,now,-rpath,$ORIGIN/../..,-rpath,$ORIGIN/../../.. -fPIC -O2 -isystem /opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/include -Wformat -Wformat-security -fstack-protector-all -D_FORTIFY_SOURCE=2 -fpic -fPIC -O2 -Wl,-z,noexecstack,-z,relro,-z,now,-rpath,$ORIGIN/../..,-rpath,$ORIGIN/../../.. -fPIC -c /var/run/palsd/9d1b9e6f-df49-49d4-bfef-16df4fa53619/tmp/tmpophyvmvj/test.c -o /var/run/palsd/9d1b9e6f-df49-49d4-bfef-16df4fa53619/tmp/tmpophyvmvj/test.o
[2025-12-31 13:11:29,205464][I][_distutils/spawn:77:spawn] icx /var/run/palsd/9d1b9e6f-df49-49d4-bfef-16df4fa53619/tmp/tmp8f2fq7zm/test.o -laio -o /var/run/palsd/9d1b9e6f-df49-49d4-bfef-16df4fa53619/tmp/tmp8f2fq7zm/a.out
[2025-12-31 13:11:29,246560][I][_distutils/spawn:77:spawn] icx /var/run/palsd/9d1b9e6f-df49-49d4-bfef-16df4fa53619/tmp/tmpatgpchkq/test.o -laio -o /var/run/palsd/9d1b9e6f-df49-49d4-bfef-16df4fa53619/tmp/tmpatgpchkq/a.out
/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/neural_compressor/utils/utility.py:44: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import parse_version
[2025-12-31 13:11:32,228] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:32,237] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:32,262] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:32,464] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:32,551] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:32,671] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:32,671] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:32,841] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:32,850] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:32,963] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:33,139] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:33,227] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:33,289658][I][ezpz/dist:2039:setup_wandb] Setting up wandb from rank=0
[2025-12-31 13:11:33,290377][I][ezpz/dist:2040:setup_wandb] Using WB_PROJECT=ezpz-hf_trainer-meta-llama-Llama-3.2-1B
[2025-12-31 13:11:33,338] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:33,463] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:33,480] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:33,494] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:33,609] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:33,629] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
[2025-12-31 13:11:33,959] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:34,200] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:34,286] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:34,294] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:34,298] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
[2025-12-31 13:11:34,305] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
wandb: Tracking run with wandb version 0.23.1
wandb: Run data is saved locally in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_131133-00uzvvx5
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run good-grass-171
wandb:  View project at https://wandb.ai/aurora_gpt/ezpz-hf_trainer-meta-llama-Llama-3.2-1B
wandb:  View run at https://wandb.ai/aurora_gpt/ezpz-hf_trainer-meta-llama-Llama-3.2-1B/runs/00uzvvx5
[2025-12-31 13:11:34,781999][I][ezpz/dist:2069:setup_wandb] wandb.run=[good-grass-171](https://wandb.ai/aurora_gpt/ezpz-hf_trainer-meta-llama-Llama-3.2-1B/runs/00uzvvx5)
[2025-12-31 13:11:34,868548][I][ezpz/dist:2112:setup_wandb] Running on machine='SunSpot'
[2025-12-31 13:11:34,874006][W][examples/hf_trainer:121:parse_args] Process rank: 0, device: xpu:0, n_gpu: 1, distributed training: True
[2025-12-31 13:11:34,875393][I][examples/hf_trainer:149:parse_args] Training/evaluation parameters TrainingArguments(
_n_gpu=1,
accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
average_tokens_across_devices=True,
batch_eval_metrics=False,
bf16=True,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
dataloader_prefetch_factor=None,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
do_eval=True,
do_predict=False,
do_train=True,
eval_accumulation_steps=None,
eval_delay=0,
eval_do_concat_batches=True,
eval_on_start=False,
eval_steps=None,
eval_strategy=no,
eval_use_gather_object=False,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[<FSDPOption.AUTO_WRAP: 'auto_wrap'>],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=True,
gradient_checkpointing_kwargs=None,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=None,
hub_revision=None,
hub_strategy=every_save,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_for_metrics=['inputs,loss'],
include_inputs_for_metrics=False,
include_num_input_tokens_seen=all,
include_tokens_per_second=True,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=5e-05,
length_column_name=length,
liger_kernel_config=None,
load_best_model_at_end=False,
local_rank=0,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=outputs/ezpz.hf_trainer/2025-12-31-131106/runs/Dec31_13-11-28_x1921c1s5b0n0,
logging_first_step=True,
logging_nan_inf_filter=True,
logging_steps=1.0,
logging_strategy=steps,
lr_scheduler_kwargs={},
lr_scheduler_type=linear,
max_grad_norm=1.0,
max_steps=100,
metric_for_best_model=None,
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
num_train_epochs=3.0,
optim=adamw_torch,
optim_args=None,
optim_target_modules=None,
output_dir=outputs/ezpz.hf_trainer/2025-12-31-131106,
overwrite_output_dir=False,
parallelism_config=None,
past_index=-1,
per_device_eval_batch_size=1,
per_device_train_batch_size=1,
prediction_loss_only=False,
project=huggingface,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=['wandb'],
restore_callback_states_from_checkpoint=False,
resume_from_checkpoint=None,
run_name=None,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=500,
save_strategy=steps,
save_total_limit=None,
seed=42,
skip_memory_metrics=True,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torch_empty_cache_steps=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
trackio_space_id=trackio,
use_cpu=False,
use_legacy_prediction_loop=False,
use_liger_kernel=False,
use_mps_device=False,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.0,
)
[INFO|configuration_utils.py:765] 2025-12-31 13:11:37,787 >> loading configuration file config.json from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/config.json
[INFO|configuration_utils.py:839] 2025-12-31 13:11:37,798 >> Model config LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "dtype": "bfloat16",
  "eos_token_id": 128001,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 16,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.3",
  "use_cache": true,
  "vocab_size": 128256
}

[INFO|tokenization_utils_base.py:2111] 2025-12-31 13:11:37,956 >> loading file tokenizer.json from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/tokenizer.json
[INFO|tokenization_utils_base.py:2111] 2025-12-31 13:11:37,957 >> loading file tokenizer.model from cache at None
[INFO|tokenization_utils_base.py:2111] 2025-12-31 13:11:37,957 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2111] 2025-12-31 13:11:37,957 >> loading file special_tokens_map.json from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/special_tokens_map.json
[INFO|tokenization_utils_base.py:2111] 2025-12-31 13:11:37,957 >> loading file tokenizer_config.json from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/tokenizer_config.json
[INFO|tokenization_utils_base.py:2111] 2025-12-31 13:11:37,957 >> loading file chat_template.jinja from cache at None
[INFO|tokenization_utils_base.py:2380] 2025-12-31 13:11:38,257 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|modeling_utils.py:1172] 2025-12-31 13:11:38,311 >> loading weights file model.safetensors from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/model.safetensors
[INFO|configuration_utils.py:986] 2025-12-31 13:11:38,313 >> Generate config GenerationConfig {
  "bos_token_id": 128000,
  "eos_token_id": 128001
}

[INFO|configuration_utils.py:941] 2025-12-31 13:12:10,482 >> loading configuration file generation_config.json from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/generation_config.json
[INFO|configuration_utils.py:986] 2025-12-31 13:12:10,482 >> Generate config GenerationConfig {
  "bos_token_id": 128000,
  "do_sample": true,
  "eos_token_id": 128001,
  "temperature": 0.6,
  "top_p": 0.9
}

[INFO|dynamic_module_utils.py:423] 2025-12-31 13:12:10,535 >> Could not locate the custom_generate/generate.py inside meta-llama/Llama-3.2-1B.
My guessed rank = 0
2025:12:31-13:12:10:(87023) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
2025:12:31-13:12:10:(87023) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
[INFO|trainer.py:699] 2025-12-31 13:12:12,555 >> max_steps is given, it will override any value given in num_train_epochs
[INFO|trainer.py:749] 2025-12-31 13:12:12,555 >> Using auto half precision backend
[INFO|trainer.py:2519] 2025-12-31 13:12:20,429 >> ***** Running training *****
[INFO|trainer.py:2520] 2025-12-31 13:12:20,429 >>   Num examples = 2,400
[INFO|trainer.py:2521] 2025-12-31 13:12:20,429 >>   Num Epochs = 9,223,372,036,854,775,807
[INFO|trainer.py:2522] 2025-12-31 13:12:20,429 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:2525] 2025-12-31 13:12:20,429 >>   Total train batch size (w. parallel, distributed & accumulation) = 24
[INFO|trainer.py:2526] 2025-12-31 13:12:20,429 >>   Gradient Accumulation steps = 1
[INFO|trainer.py:2527] 2025-12-31 13:12:20,430 >>   Total optimization steps = 100
[INFO|trainer.py:2528] 2025-12-31 13:12:20,430 >>   Number of trainable parameters = 51,492,278
[INFO|integration_utils.py:867] 2025-12-31 13:12:20,431 >> Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"
  0%|          | 0/100 [00:00<?, ?it/s]
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,209 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[WARNING|_logger.py:93] 2025-12-31 13:12:35,211 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  1%|1         | 1/100 [00:27<44:53, 27.20s/it]


{'loss': 2.5586, 'grad_norm': 6.847109794616699, 'learning_rate': 5e-05, 'epoch': 0.01, 'num_input_tokens_seen': 196608, 'train_runtime': 27.3002, 'train_tokens_per_second': 7201.711}
  1%|1         | 1/100 [00:27<44:53, 27.20s/it]
{'loss': 2.8427, 'grad_norm': 30.46836280822754, 'learning_rate': 4.9500000000000004e-05, 'epoch': 0.02, 'num_input_tokens_seen': 393216, 'train_runtime': 30.8283, 'train_tokens_per_second': 12755.034}
  2%|2         | 2/100 [00:30<21:45, 13.32s/it]
{'loss': 2.6978, 'grad_norm': 6.160351276397705, 'learning_rate': 4.9e-05, 'epoch': 0.03, 'num_input_tokens_seen': 589824, 'train_runtime': 34.3821, 'train_tokens_per_second': 17154.96}
  3%|3         | 3/100 [00:34<14:19,  8.86s/it]
{'loss': 2.7241, 'grad_norm': 8.175165176391602, 'learning_rate': 4.85e-05, 'epoch': 0.04, 'num_input_tokens_seen': 786432, 'train_runtime': 37.9205, 'train_tokens_per_second': 20738.951}
  4%|4         | 4/100 [00:37<10:48,  6.76s/it]
{'loss': 2.5838, 'grad_norm': 3.6873040199279785, 'learning_rate': 4.8e-05, 'epoch': 0.05, 'num_input_tokens_seen': 983040, 'train_runtime': 42.1184, 'train_tokens_per_second': 23339.934}
  5%|5         | 5/100 [00:42<09:14,  5.84s/it]
{'loss': 2.5221, 'grad_norm': 2.4761757850646973, 'learning_rate': 4.75e-05, 'epoch': 0.06, 'num_input_tokens_seen': 1179648, 'train_runtime': 45.6481, 'train_tokens_per_second': 25842.199}
  6%|6         | 6/100 [00:45<07:54,  5.05s/it]
{'loss': 2.5722, 'grad_norm': 2.04353404045105, 'learning_rate': 4.7e-05, 'epoch': 0.07, 'num_input_tokens_seen': 1376256, 'train_runtime': 49.1873, 'train_tokens_per_second': 27979.885}
  7%|7         | 7/100 [00:49<07:03,  4.56s/it]
{'loss': 2.5689, 'grad_norm': 2.259411573410034, 'learning_rate': 4.6500000000000005e-05, 'epoch': 0.08, 'num_input_tokens_seen': 1572864, 'train_runtime': 52.7226, 'train_tokens_per_second': 29832.818}
  8%|8         | 8/100 [00:52<06:29,  4.23s/it]
{'loss': 2.4844, 'grad_norm': 2.1371400356292725, 'learning_rate': 4.600000000000001e-05, 'epoch': 0.09, 'num_input_tokens_seen': 1769472, 'train_runtime': 56.2493, 'train_tokens_per_second': 31457.696}
  9%|9         | 9/100 [00:56<06:05,  4.01s/it]
{'loss': 2.4309, 'grad_norm': 1.866400122642517, 'learning_rate': 4.55e-05, 'epoch': 0.1, 'num_input_tokens_seen': 1966080, 'train_runtime': 60.3531, 'train_tokens_per_second': 32576.267}
 10%|#         | 10/100 [01:00<06:03,  4.04s/it]
{'loss': 2.5054, 'grad_norm': 1.7331500053405762, 'learning_rate': 4.5e-05, 'epoch': 0.11, 'num_input_tokens_seen': 2162688, 'train_runtime': 63.8948, 'train_tokens_per_second': 33847.662}
 11%|#1        | 11/100 [01:03<05:45,  3.89s/it]

# ...[clipped]...
{'loss': 2.4254, 'grad_norm': 0.8518961071968079, 'learning_rate': 2.5e-06, 'epoch': 0.96, 'num_input_tokens_seen': 18874368, 'train_runtime': 375.093, 'train_tokens_per_second': 50319.166}
 96%|#########6| 96/100 [06:15<00:14,  3.74s/it]
{'loss': 2.3401, 'grad_norm': 0.8734815716743469, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.97, 'num_input_tokens_seen': 19070976, 'train_runtime': 378.6219, 'train_tokens_per_second': 50369.452}
 97%|#########7| 97/100 [06:18<00:11,  3.68s/it]
{'loss': 2.3438, 'grad_norm': 0.9289000034332275, 'learning_rate': 1.5e-06, 'epoch': 0.98, 'num_input_tokens_seen': 19267584, 'train_runtime': 382.1676, 'train_tokens_per_second': 50416.578}
 98%|#########8| 98/100 [06:22<00:07,  3.64s/it]
{'loss': 2.3568, 'grad_norm': 0.9797356724739075, 'learning_rate': 1.0000000000000002e-06, 'epoch': 0.99, 'num_input_tokens_seen': 19464192, 'train_runtime': 385.7005, 'train_tokens_per_second': 50464.53}
 99%|#########9| 99/100 [06:25<00:03,  3.61s/it]
{'loss': 2.3369, 'grad_norm': 0.8283581137657166, 'learning_rate': 5.000000000000001e-07, 'epoch': 1.0, 'num_input_tokens_seen': 19660800, 'train_runtime': 389.2267, 'train_tokens_per_second': 50512.468}
100%|##########| 100/100 [06:29<00:00,  3.58s/it]
/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:678: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
  warnings.warn(
[INFO|trainer.py:4309] 2025-12-31 13:18:50,978 >> Saving model checkpoint to outputs/ezpz.hf_trainer/2025-12-31-131106/checkpoint-100
[INFO|configuration_utils.py:491] 2025-12-31 13:18:50,989 >> Configuration saved in outputs/ezpz.hf_trainer/2025-12-31-131106/checkpoint-100/config.json
[INFO|configuration_utils.py:757] 2025-12-31 13:18:50,995 >> Configuration saved in outputs/ezpz.hf_trainer/2025-12-31-131106/checkpoint-100/generation_config.json
[INFO|modeling_utils.py:4189] 2025-12-31 13:18:56,954 >> The model is bigger than the maximum size per checkpoint (5GB) and is going to be split in 2 checkpoint shards. You can find where each parameters has been saved in the index located at outputs/ezpz.hf_trainer/2025-12-31-131106/checkpoint-100/model.safetensors.index.json.
[INFO|tokenization_utils_base.py:2725] 2025-12-31 13:18:56,961 >> tokenizer config file saved in outputs/ezpz.hf_trainer/2025-12-31-131106/checkpoint-100/tokenizer_config.json
[INFO|tokenization_utils_base.py:2734] 2025-12-31 13:18:56,966 >> Special tokens file saved in outputs/ezpz.hf_trainer/2025-12-31-131106/checkpoint-100/special_tokens_map.json
[2025-12-31 13:18:58,330717][I][utils/fsdp_utils:131:save_fsdp_model] Saving model to outputs/ezpz.hf_trainer/2025-12-31-131106/checkpoint-100/pytorch_model_fsdp.bin
[2025-12-31 13:19:03,449014][I][utils/fsdp_utils:133:save_fsdp_model] Model saved to outputs/ezpz.hf_trainer/2025-12-31-131106/checkpoint-100/pytorch_model_fsdp.bin
[2025-12-31 13:19:17,425476][I][utils/fsdp_utils:264:save_fsdp_optimizer] Saving Optimizer state to outputs/ezpz.hf_trainer/2025-12-31-131106/checkpoint-100/optimizer.bin
[2025-12-31 13:19:25,842120][I][utils/fsdp_utils:266:save_fsdp_optimizer] Optimizer state saved in outputs/ezpz.hf_trainer/2025-12-31-131106/checkpoint-100/optimizer.bin
[INFO|trainer.py:2810] 2025-12-31 13:19:25,859 >>

Training completed. Do not forget to share your model on huggingface.co/models =)




{'train_runtime': 425.4295, 'train_samples_per_second': 5.641, 'train_steps_per_second': 0.235, 'train_tokens_per_second': 1925.584, 'train_loss': 2.489668021202087, 'epoch': 1.0, 'num_input_tokens_seen': 19660800}
100%|##########| 100/100 [07:05<00:00,  3.58s/it]
100%|##########| 100/100 [07:05<00:00,  4.25s/it]

/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:678: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
  warnings.warn(
[INFO|trainer.py:4309] 2025-12-31 13:19:27,160 >> Saving model checkpoint to outputs/ezpz.hf_trainer/2025-12-31-131106
[INFO|configuration_utils.py:491] 2025-12-31 13:19:27,171 >> Configuration saved in outputs/ezpz.hf_trainer/2025-12-31-131106/config.json
[INFO|configuration_utils.py:757] 2025-12-31 13:19:27,177 >> Configuration saved in outputs/ezpz.hf_trainer/2025-12-31-131106/generation_config.json
[INFO|modeling_utils.py:4189] 2025-12-31 13:19:33,132 >> The model is bigger than the maximum size per checkpoint (5GB) and is going to be split in 2 checkpoint shards. You can find where each parameters has been saved in the index located at outputs/ezpz.hf_trainer/2025-12-31-131106/model.safetensors.index.json.
[INFO|tokenization_utils_base.py:2725] 2025-12-31 13:19:33,139 >> tokenizer config file saved in outputs/ezpz.hf_trainer/2025-12-31-131106/tokenizer_config.json
[INFO|tokenization_utils_base.py:2734] 2025-12-31 13:19:33,143 >> Special tokens file saved in outputs/ezpz.hf_trainer/2025-12-31-131106/special_tokens_map.json
***** train metrics *****
  epoch                    =        1.0
  num_input_tokens_seen    =   19660800
  total_flos               =  4454709GF
  train_loss               =     2.4897
  train_runtime            = 0:07:05.42
  train_samples            =     726000
  train_samples_per_second =      5.641
  train_steps_per_second   =      0.235
  train_tokens_per_second  =   1925.584
[INFO|trainer.py:4643] 2025-12-31 13:19:33,298 >>
***** Running Evaluation *****
[INFO|trainer.py:4647] 2025-12-31 13:19:33,299 >>   Num examples: Unknown
[INFO|trainer.py:4648] 2025-12-31 13:19:33,299 >>   Batch size = 1
***** eval metrics *****
  epoch                   =        1.0
  eval_accuracy           =     0.5153
  eval_loss               =     2.1941
  eval_runtime            = 0:00:14.81
  eval_samples            =        100
  eval_samples_per_second =      0.337
  eval_steps_per_second   =      0.067
  num_input_tokens_seen   =   19660800
  perplexity              =     8.9716
[2025-12-31 13:19:48,219557][I][examples/hf_trainer:942:<module>] Took 500.42 seconds
wandb:
wandb: 🚀 View run good-grass-171 at:
wandb: Find logs at: ../../../../../../lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_131133-00uzvvx5/logs
My guessed rank = 7
My guessed rank = 11
My guessed rank = 6
My guessed rank = 9
My guessed rank = 1
My guessed rank = 5
My guessed rank = 3
My guessed rank = 2
My guessed rank = 8
My guessed rank = 10
My guessed rank = 23
My guessed rank = 17
My guessed rank = 20
My guessed rank = 19
My guessed rank = 18
My guessed rank = 21
My guessed rank = 4
My guessed rank = 14
My guessed rank = 15
My guessed rank = 12
My guessed rank = 16
My guessed rank = 13
My guessed rank = 22
[2025-12-31 13:19:51,674119][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2025-12-31-131951]----
[2025-12-31 13:19:51,675067][I][ezpz/launch:448:launch] Execution finished with 0.
[2025-12-31 13:19:51,675475][I][ezpz/launch:449:launch] Executing finished in 520.07 seconds.
[2025-12-31 13:19:51,675848][I][ezpz/launch:450:launch] Took 520.08 seconds to run. Exiting.
took: 8m 46s
```

</details>
