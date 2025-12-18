# HF Trainer Comparison

## Comparison

| Metric                  |         Polaris | Aurora              |
| ----------------------- | --------------: | :------------------ |
| Epoch                   |             1.0 | 2.24                |
| Perplexity              |         16.6743 | 12.7268             |
| W&B run                 | glad-moon-1[^1] | cosmic-sunset-5[^2] |
| Input tokens seen       |       6,553,600 | 19,660,800          |
| Total FLOPs             |    6,691,434 GF | 6,691,434 GF        |
| &nbsp;                  |                 |                     |
| Steps   / sec           |           0.245 | 0.334               |
| Samples / sec           |           1.958 | 8.021               |
| Tokens  / sec           |        2,004.87 | 2,737.98            |
| Tokens  / sample        |      ~ 1024[^3] | ~ 341[^4]           |
| Tokens  / step          |      ~ 8183[^5] | ~ 8197[^6]          |
| **Tokens  / sec / GPU** |      **250.61** | **114.08**          |
| &nbsp;                  |                 |                     |
| Train loss              |          3.0482 | 2.8478              |
| Train runtime           |      0:06:48.60 | 0:04:59.19          |
| Train samples           |          25,000 | 25,000              |
| &nbsp;                  |                 |                     |
| Eval accuracy           |          0.4328 | 0.4701              |
| Eval loss               |          2.8139 | 2.5437              |
| Eval runtime            |      0:00:06.26 | 0:00:09.12          |
| Eval samples            |              50 | 50                  |
| Samples / sec (eval)    |           1.118 | 0.329               |
| Steps / sec (eval)      |            0.16 | 0.11                |


[^1]: W&B Run: [glad-moon-1](https://wandb.ai/aurora_gpt/ezpz-hf_trainer--eagle-auroragpt-foremans-downloads-global_step138650/runs/k1rvbdmc)
[^2]: W&B Run: [cosmic-sunset-5](https://wandb.ai/aurora_gpt/ezpz-hf_trainer--flare-AuroraGPT-AuroraGPT-v1-Experiments-AuroraGPT-2B-public-sophiag-hf-global_step138650/runs/pqytcarn)
[^3]: Tokens per sample on Polaris:
[^4]: Tokens per sample on Aurora:
[^5]: Tokens per optimizer step on Polaris:
      - ~ **8183** = (2004.87 / 0.245) \[`tokens_per_second` / `steps_per_second`\]
[^6]: Tokens per optimizer step on Aurora:
      - ~ **8197** = (2737.984 / 0.334) \[`tokens_per_second` / `steps_per_second`\]

## Details

> [!NOTE]
> - **Samples per optimizer step**:
>   - Polaris: train_samples_per_second / train_steps_per_second = 1.958 / 0.245 ≈ 8.0 samples/step ✅ matches global batch 8
>   - Aurora: 8.021 / 0.334 ≈ 24.0 samples/step ✅ matches global batch 24
>
> - **Tokens per optimizer step**:
>   - Polaris: train_tokens_per_second / train_steps_per_second = 2004.87 / 0.245 ≈ 8183 tokens/step
>   - Aurora: 2737.984 / 0.334 ≈ 8197 tokens/step

So both runs are doing ~8192 tokens per optimizer step (close enough to “same”), which means:

Your tokens/sec difference is almost entirely explained by steps/sec.

Because:

$$
\text{tokens/sec} = (\text{tokens/step}) \times (\text{steps/sec})
$$

The fair comparison (given your setup)

Since tokens/step is basically equal, compare steps/sec (or step time):

- Polaris: 0.245 steps/sec → step time ≈ 4.08 s/step
- Aurora: 0.334 steps/sec → step time ≈ 2.99 s/step

Aurora is ~1.36× faster per optimizer step (0.334 / 0.245), and therefore
~1.36× faster in tokens/sec, for this exact training configuration.

If your real question is “which system is more efficient per device?”

Then normalize by device count (this will penalize Aurora because you’re using
3× more devices):

- **Steps/sec/device**:
  - Polaris: 0.245 / 8 = 0.0306
  - Aurora: 0.334 / 24 = 0.0139

  So per device, Polaris is ~2.2× “better” on this metric.

But that’s not a hardware truth statement; it’s mostly a scaling regime
statement: 24-way data parallel will usually lose per-device efficiency to
comms/overheads vs 8-way, unless you increase work per device (bigger
microbatch, longer seq, etc.).

What you should report (so nobody can nitpick you)

- Use these two simultaneously:

  1. Time-to-train / throughput at chosen scale (what you actually care about operationally)
     - Aurora: 1.36× higher steps/sec and tokens/sec here.
  1. Scaling efficiency (how hard you’re paying for using more devices)
     - Per-device efficiency ratio: (0.334/24) / (0.245/8) ≈ 0.45
     - Meaning: Aurora at 24 devices is delivering ~45% of the per-device step throughput you got on Polaris at 8 devices.

- One more blunt point:

  Even though tokens/step match, your tokens/sample differ a lot:

  - **Polaris**: ~ `8183 /  8 ≈ 1024` \[tokens/sample\]
  - **Aurora**:  ~ `8197 / 24 ≈  341` \[tokens/sample\]

  So you are not feeding the same “sample” definition (sequence length /
  packing / truncation).

  That’s fine as long as you compare per-step or per-token, not per-sample.

  If you want an airtight apples-to-apples hardware bakeoff: 

  - **fix seq_len and packing*** so tokens/sample matches, then sweep DP size
    (8 vs 24) on each system and plot step time vs devices.

    That will tell you whether Aurora is “worse per device” because of comms,
    kernel maturity, input pipeline, or just the scaling point you chose.

## Running on Aurora

```bash
#[aurora_frameworks-2025.2.0](ezpz-distributed-metrics-aurora_frameworks-2025.2.0)
#[12/18/25,13:21:37][x4310c7s4b0n0][/f/A/A/E/A/t/s/ezpz-distributed-metrics][ distributed-metrics][$?] [1m11s]
; ckpt=/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650 ; ezpz launch python3 -m ezpz.examples.hf_trainer --streaming --dataset_name=stanfordnlp/imdb --model_name_or_path"${ckpt}" --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --include-tokens-per-second=true --max-steps=100 --include-num-input-tokens-seen=true --optim=adamw_torch --logging-first-step --include-for-metrics='inputs,loss' --max-eval-samples=50 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=8192 --gradient_checkpointing=true --fsdp=auto_wrap --output_dir=$(tstamp)
```

- <details closed><sumamry>Output:</sumamry>

  ```bash
  # [2025-12-18 13:21:50,091003][I][ezpz/launch:378:launch] ----[🍋 ezpz.launch][started][2025-12-18-132150]----
  # [2025-12-18 13:21:51,231621][I][ezpz/launch:396:launch] Job ID: 8219131
  # [2025-12-18 13:21:51,232430][I][ezpz/launch:397:launch] nodelist: ['x4310c7s4b0n0', 'x4418c6s1b0n0']
  # [2025-12-18 13:21:51,232855][I][ezpz/launch:398:launch] hostfile: /var/spool/pbs/aux/8219131.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
  # [2025-12-18 13:21:51,233490][I][ezpz/pbs:329:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
  # [2025-12-18 13:21:51,234263][I][ezpz/launch:354:build_executable] Building command to execute by piecing together:
  # [2025-12-18 13:21:51,234692][I][ezpz/launch:355:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8219131.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
  [2025-12-18 13:21:51,235394][I][ezpz/launch:356:build_executable] (2.) cmd_to_launch: python3 -m ezpz.examples.hf_trainer --streaming --dataset_name=stanfordnlp/imdb --model_name_or_path /flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650 --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --include-tokens-per-second=true --max-steps=100 --include-num-input-tokens-seen=true --optim=adamw_torch --logging-first-step --include-for-metrics=inputs,loss --max-eval-samples=50 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=8192 --gradient_checkpointing=true --fsdp=auto_wrap --output_dir=2025-12-18-132143
  # [2025-12-18 13:21:51,237243][I][ezpz/launch:412:launch] Took: 1.22 seconds to build command.
  # [2025-12-18 13:21:51,237645][I][ezpz/launch:413:launch] Executing:
  mpiexec
    --envall
    --np=24
    --ppn=12
    --hostfile=/var/spool/pbs/aux/8219131.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    --no-vni
    --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
    python3
    -m
    ezpz.examples.hf_trainer
    --streaming
    --dataset_name=stanfordnlp/imdb
    --model_name_or_path
    /flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650
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
    --max-eval-samples=50
    --per_device_train_batch_size=1
    --per_device_eval_batch_size=1
    --block_size=8192
    --gradient_checkpointing=true
    --fsdp=auto_wrap
    --output_dir=2025-12-18-132143
  [2025-12-18 13:21:51,240032][I][ezpz/launch:213:get_aurora_filters] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG
  [2025-12-18 13:21:51,240580][I][ezpz/launch:420:launch] Execution started @ 2025-12-18-132151...
  [2025-12-18 13:21:51,241009][I][ezpz/launch:421:launch] ----[🍋 ezpz.launch][stop][2025-12-18-132151]----
  [2025-12-18 13:21:51,241473][I][ezpz/launch:132:run_command] Caught 24 filters
  [2025-12-18 13:21:51,241853][I][ezpz/launch:133:run_command] Running command:
   mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8219131.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 -m ezpz.examples.hf_trainer --streaming --dataset_name=stanfordnlp/imdb --model_name_or_path /flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650 --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --include-tokens-per-second=true --max-steps=100 --include-num-input-tokens-seen=true --optim=adamw_torch --logging-first-step --include-for-metrics=inputs,loss --max-eval-samples=50 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=8192 --gradient_checkpointing=true --fsdp=auto_wrap --output_dir=2025-12-18-132143
  [2025-12-18 13:22:09,760765][I][ezpz/dist:1926:setup_wandb] Using WB_PROJECT=ezpz-hf_trainer--flare-AuroraGPT-AuroraGPT-v1-Experiments-AuroraGPT-2B-public-sophiag-hf-global_step138650
  [2025-12-18 13:22:11,079430][I][ezpz/dist:1955:setup_wandb] wandb.run=[cosmic-sunset-5](https://wandb.ai/aurora_gpt/ezpz-hf_trainer--flare-AuroraGPT-AuroraGPT-v1-Experiments-AuroraGPT-2B-public-sophiag-hf-global_step138650/runs/pqytcarn)
  [WARNING|trainer.py:982] 2025-12-18 13:23:46,860 >> The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly,being updated with the tokenizers values. Updated tokens: {'eos_token_id': 1, 'bos_token_id': 2, 'pad_token_id': 0}.
  [INFO|trainer.py:2519] 2025-12-18 13:23:51,164 >> ***** Running training *****
  [INFO|trainer.py:2520] 2025-12-18 13:23:51,164 >>   Num examples = 2,400
  [INFO|trainer.py:2521] 2025-12-18 13:23:51,164 >>   Num Epochs = 9,223,372,036,854,775,807
  [INFO|trainer.py:2522] 2025-12-18 13:23:51,164 >>   Instantaneous batch size per device = 1
  [INFO|trainer.py:2525] 2025-12-18 13:23:51,164 >>   Total train batch size (w. parallel, distributed & accumulation) = 24
  [INFO|trainer.py:2526] 2025-12-18 13:23:51,165 >>   Gradient Accumulation steps = 1
  [INFO|trainer.py:2527] 2025-12-18 13:23:51,165 >>   Total optimization steps = 100
  [INFO|trainer.py:2528] 2025-12-18 13:23:51,165 >>   Number of trainable parameters = 82,752,264
  [INFO|integration_utils.py:867] 2025-12-18 13:23:51,171 >> Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"
    0%|          | 0/100 [00:00<?, ?it/s]
  # [...clipped...]
  {'loss': 2.6031, 'grad_norm': 1.8801486492156982, 'learning_rate': 5.000000000000001e-07, 'epoch': 2.24, 'num_input_tokens_seen': 19660800, 'train_runtime': 252.007, 'train_tokens_per_second': 78016.894}
  100%|##########| 100/100 [04:11<00:00,  2.27s/it]
  # [...clipped...]
  [INFO|trainer.py:4309] 2025-12-18 13:28:05,079 >> Saving model checkpoint to 2025-12-18-132143/checkpoint-100
  {'train_runtime': 299.1983, 'train_samples_per_second': 8.021, 'train_steps_per_second': 0.334, 'train_tokens_per_second': 2737.984, 'train_loss': 2.8478338932991027, 'epoch': 2.24, 'num_input_tokens_seen': 19660800}
  100%|##########| 100/100 [04:59<00:00,  2.27s/it]
  [INFO|trainer.py:4309] 2025-12-18 13:28:52,199 >> Saving model checkpoint to 2025-12-18-132143
  ***** train metrics *****
    epoch                    =       2.24
    num_input_tokens_seen    =   19660800
    total_flos               =  6691434GF
    train_loss               =     2.8478
    train_runtime            = 0:04:59.19
    train_samples            =      25000
    train_samples_per_second =      8.021
    train_steps_per_second   =      0.334
    train_tokens_per_second  =   2737.984
  [INFO|trainer.py:4643] 2025-12-18 13:29:00,076 >>
  ***** Running Evaluation *****
  [INFO|trainer.py:4647] 2025-12-18 13:29:00,077 >>   Num examples: Unknown
  [INFO|trainer.py:4648] 2025-12-18 13:29:00,077 >>   Batch size = 1
  ***** eval metrics *****
    epoch                   =       2.24
    eval_accuracy           =     0.4701
    eval_loss               =     2.5437
    eval_runtime            = 0:00:09.12
    eval_samples            =         50
    eval_samples_per_second =      0.329
    eval_steps_per_second   =       0.11
    num_input_tokens_seen   =   19660800
    perplexity              =    12.7268
  wandb:
  wandb: 🚀 View run cosmic-sunset-5 at:
  wandb: Find logs at: ../../../../../../../../lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/ezpz-distributed-metrics/wandb/run-20251218_132210-pqytcarn/logs
  ```

</details>


## Running on Polaris


```bash
(2025-09-25/base) (ezpz-distributed-metrics-mconda3)
#[/e/A/f/p/s/ezpz-distributed-metrics][🌱 distributed-metrics][🤷✓] [⏱️ 1m28s]
#[12/18/25 @ 13:32:34][x3006c0s1b0n0]
; ckpt=/eagle/AuroraGPT/foremans/Downloads/global_step138650 ; ezpz launch python3 -m ezpz.examples.hf_trainer --streaming --dataset_name=stanfordnlp/imdb --model_name_or_path "${ckpt}" --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --include-tokens-per-second=true --max-steps=100 --include-num-input-tokens-seen=true --optim=adamw_torch --logging-first-step --include-for-metrics='inputs,loss' --max-eval-samples=50 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=8192 --gradient_checkpointing=true --fsdp=auto_wrap --output_dir=$(tstamp)
```


- <details closed><sumamry>Output:</sumamry>

  ```bash
  [2025-12-18 13:32:45,480638][i][ezpz/launch:378:launch] ----[🍋 ezpz.launch][started][2025-12-18-133245]----
  [2025-12-18 13:32:46,548182][i][ezpz/launch:396:launch] job id: 6815207
  [2025-12-18 13:32:46,549238][i][ezpz/launch:397:launch] nodelist: ['x3006c0s1b0n0', 'x3006c0s25b0n0']
  [2025-12-18 13:32:46,549674][i][ezpz/launch:398:launch] hostfile: /var/spool/pbs/aux/6815207.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
  [2025-12-18 13:32:46,550353][i][ezpz/pbs:329:get_pbs_launch_cmd] ✅ using [8/8] gpus [2 hosts] x [4 gpu/host]
  [2025-12-18 13:32:46,551713][i][ezpz/launch:354:build_executable] building command to execute by piecing together:
  [2025-12-18 13:32:46,552123][i][ezpz/launch:355:build_executable] (1.) launch_cmd: mpiexec --envall --np=8 --ppn=4 --hostfile=/var/spool/pbs/aux/6815207.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind=depth --depth=8
  [2025-12-18 13:32:46,552678][i][ezpz/launch:356:build_executable] (2.) cmd_to_launch: python3 -m ezpz.examples.hf_trainer --streaming --dataset_name=stanfordnlp/imdb --model_name_or_path /eagle/auroragpt/foremans/downloads/global_step138650 --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1--include-tokens-per-second=true --max-steps=100 --include-num-input-tokens-seen=true --optim=adamw_torch --logging-first-step --include-for-metrics=inputs,loss --max-eval-samples=50 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=8192 --gradient_checkpointing=true --fsdp=auto_wrap --output_dir=2025-12-18-133234
  [2025-12-18 13:32:46,554200][i][ezpz/launch:412:launch] took: 1.43 seconds to build command.
  [2025-12-18 13:32:46,554592][i][ezpz/launch:413:launch] executing:
  mpiexec
  --envall
  --np=8
  --ppn=4
  --hostfile=/var/spool/pbs/aux/6815207.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
  --cpu-bind=depth
  --depth=8
  python3
  -m
  ezpz.examples.hf_trainer
  --streaming
  --dataset_name=stanfordnlp/imdb
  --model_name_or_path
  /eagle/auroragpt/foremans/downloads/global_step138650
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
  --max-eval-samples=50
  --per_device_train_batch_size=1
  --per_device_eval_batch_size=1
  --block_size=8192
  --gradient_checkpointing=true
  --fsdp=auto_wrap
  --output_dir=2025-12-18-133234
  [2025-12-18 13:32:46,557424][i][ezpz/launch:420:launch] execution started @ 2025-12-18-133246...
  [2025-12-18 13:32:46,557890][i][ezpz/launch:421:launch] ----[🍋 ezpz.launch][stop][2025-12-18-133246]----
  [2025-12-18 13:32:46,558395][i][ezpz/launch:133:run_command] running command:
  mpiexec --envall --np=8 --ppn=4 --hostfile=/var/spool/pbs/aux/6815207.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind=depth --depth=8 python3 -m ezpz.examples.hf_trainer --streaming --dataset_name=stanfordnlp/imdb --model_name_or_path /eagle/auroragpt/foremans/downloads/global_step138650 --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --include-tokens-per-second=true --max-steps=100 --include-num-input-tokens-seen=true --optim=adamw_torch --logging-first-step --include-for-metrics=inputs,loss --max-eval-samples=50 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=8192 --gradient_checkpointing=true --fsdp=auto_wrap --output_dir=2025-12-18-133234
  [2025-12-18 13:33:46,605589][i][ezpz/dist:1926:setup_wandb] using wb_project=ezpz-hf_trainer--eagle-auroragpt-foremans-downloads-global_step138650
  [2025-12-18 13:33:54,462981][i][ezpz/dist:1955:setup_wandb] wandb.run=[glad-moon-1](https://wandb.ai/aurora_gpt/ezpz-hf_trainer--eagle-auroragpt-foremans-downloads-global_step138650/runs/k1rvbdmc)
  [info|trainer.py:2519] 2025-12-18 13:34:42,086 >> ***** running training *****
  [info|trainer.py:2520] 2025-12-18 13:34:42,087 >>   num examples = 800
  [info|trainer.py:2521] 2025-12-18 13:34:42,087 >>   num epochs = 9,223,372,036,854,775,807
  [info|trainer.py:2522] 2025-12-18 13:34:42,087 >>   instantaneous batch size per device = 1
  [info|trainer.py:2525] 2025-12-18 13:34:42,087 >>   total train batch size (w. parallel, distributed & accumulation) = 8
  [info|trainer.py:2526] 2025-12-18 13:34:42,087 >>   gradient accumulation steps = 1
  [info|trainer.py:2527] 2025-12-18 13:34:42,087 >>   total optimization steps = 100
  [info|trainer.py:2528] 2025-12-18 13:34:42,087 >>   number of trainable parameters = 248,256,768
  [info|integration_utils.py:867] 2025-12-18 13:34:43,444 >> automatic weights & biases logging enabled, to disable set os.environ["wandb_disabled"] = "true"
  0%|          | 0/100 [00:00<?, ?it/s]
  # [...clipped...]
  {'loss': 2.9705, 'grad_norm': 1.2514474391937256, 'learning_rate': 5.000000000000001e-07, 'epoch': 1.0, 'num_input_tokens_seen': 6553600, 'train_runtime': 356.0234, 'train_tokens_per_second': 18407.78}
  100%|##########| 100/100 [05:54<00:00,  3.45s/it]
  # [...clipped...]
  [info|trainer.py:4309] 2025-12-18 13:40:42,445 >> saving model checkpoint to 2025-12-18-133234/checkpoint-100
  {'train_runtime': 408.6051, 'train_samples_per_second': 1.958, 'train_steps_per_second': 0.245, 'train_tokens_per_second': 2004.87, 'train_loss': 3.048191022872925, 'epoch': 1.0, 'num_input_tokens_seen': 6553600}
  100%|##########| 100/100 [06:47<00:00,  3.45s/it]
  100%|##########| 100/100 [06:47<00:00,  4.07s/it]
  [info|trainer.py:4309] 2025-12-18 13:41:34,895 >> saving model checkpoint to 2025-12-18-133234
  ***** train metrics *****
  epoch                    =        1.0
  num_input_tokens_seen    =    6553600
  total_flos               =  6691434gf
  train_loss               =     3.0482
  train_runtime            = 0:06:48.60
  train_samples            =      25000
  train_samples_per_second =      1.958
  train_steps_per_second   =      0.245
  train_tokens_per_second  =    2004.87
  [info|trainer.py:4643] 2025-12-18 13:41:43,747 >>
  ***** running evaluation *****
  [info|trainer.py:4647] 2025-12-18 13:41:43,747 >>   num examples: unknown
  [info|trainer.py:4648] 2025-12-18 13:41:43,747 >>   batch size = 1
  ***** eval metrics *****
  epoch                   =        1.0
  eval_accuracy           =     0.4328
  eval_loss               =     2.8139
  eval_runtime            = 0:00:06.26
  eval_samples            =         50
  eval_samples_per_second =      1.118
  eval_steps_per_second   =       0.16
  num_input_tokens_seen   =    6553600
  perplexity              =    16.6743
  wandb:
  wandb: 🚀 view run glad-moon-1 at:
  wandb: find logs at: ../../../../../../lus/eagle/projects/auroragpt/foremans/projects/saforem2/ezpz-distributed-metrics/wandb/run-20251218_133347-k1rvbdmc/logs
  ```

</details>
