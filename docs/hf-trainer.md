
# Language Model Training with 🍋 `ezpz` and 🤗 HF Trainer

The [`src/ezpz/hf_trainer.py`](/src/ezpz/hf_trainer.py) module provides a
mechanism for distributed training with 🤗 [huggingface /
transformers](https://github.com/huggingface/transformers).

In particular, it allows for distributed training using the
[`transformers.Trainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer)
object with **_any_**[^any] (compatible) combination of
{[`models`](https://huggingface.co/models),
[`datasets`](https://huggingface.co/datasets)}.

[^any]: See the full list of supported models at:
    [https://hf.co/models?filter=text-generation](https://huggingface.co/models?filter=text-generation)

Additionally, [DeepSpeed](https://github.com/deepspeedai/deepspeed) is fully
supported and can be configured by specifying
`--deepspeed=/path/to/deepspeed_config.json` in the command line.

## 🐣 Getting Started

1. 🏡 Setup environment (on ANY {Intel, NVIDIA, AMD} accelerator)

    ```bash
    source <(curl -s 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh')
    ezpz_setup_env
    ```

1. 📦 Install dependencies:

    1. Install 🍋 `ezpz` (from GitHub):

        ```bash
        python3 -m pip install -e "git+https://github.com/saforem2/ezpz" --require-virtualenv
        ```

    1. Update {`transformers`, `evaluate`}:

        ```bash
        python3 -m pip install --upgrade transformers evaluate
        ```

1. 🚀 Launch training:

    ```bash
    launch python3 -m ezpz.hf_trainer \
      --dataset_name stanfordnlp/imdb \                 # Example dataset
      --model_name_or_path meta-llama/Llama-3.2-1B \    # Example model
      --bf16 \                                          # TrainingArguments
      --do_train \
      --block_size=8 \
      --gradient_checkpointing \
      --per_device_train_batch_size=1 \
      --deepspeed=ds_configs/zero_stage1_config.json \
      --output_dir=trainer_output-$(date "+%Y-%m-%d-%H%M%S")
    ```
