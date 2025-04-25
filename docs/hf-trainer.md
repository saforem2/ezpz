# Language Model Training with 🍋 `ezpz` and 🤗 HF Trainer

```bash
#!/bin/bash

###### CONFIG
ACCEPTED_HOSTS="/root/.hag_accepted.conf"
BE_VERBOSE=false

if [ "$UID" -ne 0 ]
then
 echo "Superuser rights required"
 exit 2
fi

genApacheConf(){
 echo -e "# Host ${HOME_DIR}$1/$2 :"
}

echo '"quoted"' | tr -d \" > text.txt
```

The
[`src/ezpz/hf_trainer.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/hf_trainer.py)
module provides a mechanism for distributed training with 🤗 [huggingface /
transformers](https://github.com/huggingface/transformers).

In particular, it allows for distributed training using the
[`transformers.Trainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer)
object with **_any_**[^any] (compatible) combination of
{[`models`](https://huggingface.co/models),
[`datasets`](https://huggingface.co/datasets)}.

[^any]:
    See the full list of supported models at:
    [https://hf.co/models?filter=text-generation](https://huggingface.co/models?filter=text-generation)

## 🐣 Getting Started

1. 🏡 Setup environment (on ANY {Intel, NVIDIA, AMD} accelerator)

   ```bash
   source <(curl -s 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh')
   ezpz_setup_env
   ```

1. 📦 Install dependencies:

   1. Install 🍋 `ezpz` (from GitHub):

      ```bash
      python3 -m pip install "git+https://github.com/saforem2/ezpz" --require-virtualenv
      ```

   1. Update {`transformers`, `evaluate`}:

      ```bash
      python3 -m pip install --upgrade transformers evaluate
      ```

1. 🚀 Launch training:

   ```bash
   python3 -m ezpz.launch -m ezpz.hf_trainer \  # (1)
       --dataset_name stanfordnlp/imdb \
       --model_name_or_path meta-llama/Llama-3.2-1B \
       --bf16 \
       --do_train \
       --report-to=wandb \
       --logging-steps=1 \
       --include-tokens-per-second=true \
       --auto-find-batch-size=true \
       --output_dir=outputs/ezpz-hf-trainer/$(date "+%Y-%m-%d-%H%M%S") \
       --ddp-backend=$(echo "$([ $(ezpz_get_machine_name)=="aurora" ] && echo "ccl" || echo "nccl")")
   ```

   1. 🪄 <b>Magic</b>:

   Behind the scenes, this will 🪄 _automagically_ determine
   the specifics of the running job, and use this information to
   construct (and subsequently run) the appropriate:

   ```shell
   mpiexec <mpi-args> $(which python3) <cmd-to-launch>
   ```

   across all of our available accelerators.

   - <details closed><summary>➕ <b>Tip</b>:</summary>

     Call:

     ```bash
     python3 -m ezpz.hf_trainer --help
     ```

     to see the full list of supported arguments.

     In particular, _**any**_ `transformers.TrainingArguments` _should_ be supported.

     </details>

## 🚀 DeepSpeed Support

Additionally, [DeepSpeed](https://github.com/deepspeedai/deepspeed) is fully
supported and can be configured by specifying the path to a compatible
[DeepSpeed config json file](https://www.deepspeed.ai/docs/config-json/), e.g.:

1. Build a DeepSpeed config:

   ```bash
   python3 -c 'import ezpz; ezpz.utils.write_deepspeed_zero12_auto_config(zero_stage=2)'
   ```

2. Train:

   ```bash
   python3 -m ezpz.launch -m ezpz.hf_trainer \
     --dataset_name stanfordnlp/imdb \
     --model_name_or_path meta-llama/Llama-3.2-1B \
     --bf16 \
     --do_train \
     --report-to=wandb \
     --logging-steps=1 \
     --include-tokens-per-second=true \
     --auto-find-batch-size=true \
     --deepspeed=ds_configs/deepspeed_zero2_auto_config.json
   ```

> 😎 2 ez
