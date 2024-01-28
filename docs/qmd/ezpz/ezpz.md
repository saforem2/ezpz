# Starting Up Distributed Training
Sam Foreman
2024-01-28

## Initialization Times

<!-- ::: {.callout-info title='[ Startup Times (Perlmutter)]{.dim-text}' collapse="True"} -->

> [!TIP]
>
> ### <span class="dim-text"><span class="quarto-shortcode__" data-is-shortcode="1" data-raw="{{&lt; iconify material-symbols attach-email-outline-rounded &gt;}}"><span class="quarto-shortcode__-param" data-is-shortcode="1" data-value="iconify" data-raw="iconify"></span> <span class="quarto-shortcode__-param" data-is-shortcode="1" data-value="material-symbols" data-raw="material-symbols"></span> <span class="quarto-shortcode__-param" data-is-shortcode="1" data-value="attach-email-outline-rounded" data-raw="attach-email-outline-rounded"></span></span> Application Startup Time</span>
>
> - From Tanima:
>
>   > Hi Sam and Corey,
>   >
>   > Thanks for your comments on measuring the application start up
>   > time last week.
>   >
>   > Typically, we report the throughput performance after the start-up
>   > and warm-up during the “steady” state of the training.
>   >
>   > We have a few follow-up questions so that we establish a
>   > methodology to address the issue brought up by Argonne.
>   >
>   > 1.  We can set a few timestamps in the model scripts and job
>   >     scripts used for the queue submission: Job script:
>   >
>   >     ``` bash
>   >     Time stamp A:  
>   >     <actual python command using mpiexec>
>   >
>   >     Inside the model script:  
>   >     main()  
>   >     Timestamp B:  
>   >     [...]
>   >     Timestamp C:  
>   >     First training steps and onwards.  
>   >     ```
>   >
>   >     By startup time, do you mean measuring time difference between
>   >     A and C or B and C?
>
>   > 2.  Will the measurement methodology be the same for distributed
>   >     training?  
>   >     For examples, we can measure the start-up time for the
>   >     `rank0`?
>
>   > 3.  If we need to report the startup time for the DL applications,
>   >     do we need to collect measurements using the actual Aurora NRE
>   >     workloads or some small benchmarking test cases?
>   >
>   >     For example, we can try to recreate the typical start-up
>   >     scenarios, like library imports, and measure those separately
>   >     as shown below.
>   >
>   >     ``` bash
>   >     Job script:
>   >     Time stamp A:
>   >     <actual python command using mpiexec>
>   >
>   >     Time stamp B:
>   >      import torch
>   >     Time stamp C
>   >     import IPEX
>   >     Time stamp D
>   >     Etc...
>   >     ```
>   >
>   >     If you have any other scenarios, please feel free to suggest.
>   >
>   > Thanks, Tanima.

1.  In [Measuring / Calculating Startup Time](#sec-measurements),I
    provide a summary of how the startup time is identified and
    calculated.

2.  I’m not sure exactly I understand

    > Will the measurement methodology be the same for distributed
    > training? For examples, we can measure the start-up time for the
    > rank0?

    The startup time is being measured for distributed training (logs
    only created on `RANK = 0`)

3.  I discuss in [Minimal Working Example](#minimal-working-example) a
    minimal example that can be used to measure the startup times.

    - This is using a library I’ve been working on,
      [`ezpz`](https://github.com/saforem2/ezpz) that is designed to
      help simplify the process of setting up / initializing distributed
      training across many GPUs.

### Measuring / Calculating Startup Time

- The startup timing was identified by parsing the logfiles from
  existing runs and calculating the difference
  $\delta t = t_{1} - t_{0}$,

  - $t_{0}$ is the time stamp at the *very* beginning of the shell
    script (defined
    [here](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/657bb3e312e793d7b503f475b59e44c1aee44205/ALCF/train-gpt3.sh#L3))
    which then launches `mpiexec <mpi-args> python3 [...]`.

    - $t_{0}$ appears in the logfile as:

      ``` bash
      Job started at: 2023-11-02-183323 on x3004c0s13b0n0
      ```

  - $t_{1}$ is identified as the timestamp associated with the
    completion of the first training step

    - $t_{1}$ appears in the logfile as:

      ``` bash
      [2023-11-02 18:34:13,122] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[0.0, 0.0], mom=[(0.9, 0.999), (0.9, 0.999)]
      ```

- Below is an example of the bash script use to parse the logfiles and
  identify these timestamps:

  ``` bash
    $ for f in $(tail -5 logfiles) ; do echo $f; cat $f | grep -E "Job started|step=0\," | uniq ; echo "\n" ; done
    /lus/grand/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/outputs/gpt_actCkpt_GPT1T_4L_z1_seqlen2048_mp8_pp2_sp1_nl4_hs25600_gb16_mb1/logs/foremans-x3004c0s13b0n0-nhosts4-ngpu16-2023-11-02-183323.log
    Job started at: 2023-11-02-183323 on x3004c0s13b0n0
    [2023-11-02 18:34:13,122] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[0.0, 0.0], mom=[(0.9, 0.999), (0.9, 0.999)]

    /lus/grand/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/outputs/gpt_SP_actCkpt_GPT125M_z0_seqlen2048_mp16_pp1_sp1_nl12_hs768_gb1_mb1/logs/foremans-x3015c0s37b0n0-nhosts4-ngpu16-2023-11-02-184240.log
    Job started at: 2023-11-02-184240 on x3015c0s37b0n0

    /lus/grand/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/outputs/gpt_SP_actCkpt_GPT125M_z0_seqlen2048_mp16_pp1_sp1_nl12_hs768_gb1_mb1/logs/foremans-x3015c0s37b0n0-nhosts4-ngpu16-2023-11-02-184259.log
    Job started at: 2023-11-02-184259 on x3015c0s37b0n0
    [2023-11-02 18:43:23,385] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[0.0, 0.0], mom=[(0.9, 0.999), (0.9, 0.999)]

    /lus/grand/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/outputs/gpt_SP_actCkpt_GPT125M_z0_seqlen2048_mp16_pp1_sp1_nl12_hs768_gb1_mb1/logs/foremans-x3004c0s13b0n0-nhosts4-ngpu16-2023-11-02-184407.log
    Job started at: 2023-11-02-184407 on x3004c0s13b0n0
    [2023-11-02 18:44:32,804] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[0.0, 0.0], mom=[(0.9, 0.999), (0.9, 0.999)]

    /lus/grand/projects/datascience/foremans/locations/polaris/projects/argonne-lcf/Megatron-DeepSpeed/outputs/gpt_actCkpt_GPT1T_4L_z1_seqlen2048_mp8_pp2_sp1_nl4_hs25600_gb16_mb2/logs/foremans-x3108c0s25b1n0-nhosts2-ngpu8-2023-11-02-192739.log
    Job started at: 2023-11-02-192739 on x3108c0s25b1n0
  ```

> [!TIP]
>
> ### <span class="dim-text"><span class="quarto-shortcode__" data-is-shortcode="1" data-raw="{{&lt; iconify octicon stopwatch-16 &gt;}}"><span class="quarto-shortcode__-param" data-is-shortcode="1" data-value="iconify" data-raw="iconify"></span> <span class="quarto-shortcode__-param" data-is-shortcode="1" data-value="octicon" data-raw="octicon"></span> <span class="quarto-shortcode__-param" data-is-shortcode="1" data-value="stopwatch-16" data-raw="stopwatch-16"></span></span> Startup Times (Perlmutter)</span>
>
> |                           \*\*\*\*                           | **model_size** | **world_size** |     **start**     |     **stop**      | **t0** | **t1** | **dt** |
> |:------------------------------------------------------------:|:--------------:|:--------------:|:-----------------:|:-----------------:|:------:|:------:|:------:|
> |   `foremans-nid008217-nhosts2-ngpu8-2023-10-05-191101.log`   |    GPT1T_1L    |       8        | 2023-10-05-191101 | 2023-10-05-191215 | 191101 | 191215 |  114   |
> |   `foremans-nid008217-nhosts2-ngpu8-2023-10-05-191400.log`   |    GPT1T_1L    |       8        | 2023-10-05-191400 | 2023-10-05-191511 | 191400 | 191511 |  111   |
> |   `foremans-nid008217-nhosts2-ngpu8-2023-10-05-191707.log`   |    GPT1T_1L    |       8        | 2023-10-05-191707 | 2023-10-05-191817 | 191707 | 191817 |  110   |
> |   `foremans-nid008553-nhosts2-ngpu8-2023-10-15-114506.log`   |    GPT1T_2L    |       8        | 2023-10-15-114506 | 2023-10-15-114616 | 114506 | 114616 |  110   |
> |   `foremans-nid008572-nhosts2-ngpu8-2023-10-15-133531.log`   |    GPT2_7B     |       8        | 2023-10-15-133531 | 2023-10-15-133745 | 133531 | 133745 |  214   |
> |   `foremans-nid008572-nhosts2-ngpu8-2023-10-15-135041.log`   |    GPT2_7B     |       8        | 2023-10-15-135041 | 2023-10-15-135255 | 135041 | 135255 |  214   |
> |   `foremans-nid008572-nhosts2-ngpu8-2023-10-15-140806.log`   |    GPT2_7B     |       8        | 2023-10-15-140806 | 2023-10-15-141236 | 140806 | 141236 |  430   |
> |   `foremans-nid008572-nhosts2-ngpu8-2023-10-15-143120.log`   |    GPT2_7B     |       8        | 2023-10-15-143120 | 2023-10-15-143655 | 143120 | 143655 |  535   |
> |   `foremans-nid008268-nhosts2-ngpu8-2023-10-15-154337.log`   |    GPT2_7B     |       8        | 2023-10-15-154337 | 2023-10-15-154446 | 154337 | 154446 |  109   |
> |   `foremans-nid008268-nhosts2-ngpu8-2023-10-15-154943.log`   |    GPT1T_1L    |       8        | 2023-10-15-154943 | 2023-10-15-155317 | 154943 | 155317 |  374   |
> |   `foremans-nid008268-nhosts2-ngpu8-2023-10-15-162315.log`   |    GPT1T_1L    |       8        | 2023-10-15-162315 | 2023-10-15-162441 | 162315 | 162441 |  126   |
> |    `foremans-login12-nhosts2-ngpu8-2023-10-15-180714.log`    |    GPT2_7B     |       8        | 2023-10-15-180714 | 2023-10-15-180805 | 180714 | 180805 |   91   |
> |    `foremans-login12-nhosts2-ngpu8-2023-10-15-181733.log`    |    GPT2_7B     |       8        | 2023-10-15-181733 | 2023-10-15-181834 | 181733 | 181834 |  101   |
> |    `foremans-login12-nhosts2-ngpu8-2023-10-15-182228.log`    |    GPT1T_1L    |       8        | 2023-10-15-182228 | 2023-10-15-183031 | 182228 | 183031 |  803   |
> |    `foremans-login12-nhosts2-ngpu8-2023-10-15-183345.log`    |    GPT1T_2L    |       8        | 2023-10-15-183345 | 2023-10-15-183750 | 183345 | 183750 |  405   |
> |    `foremans-login12-nhosts2-ngpu8-2023-10-15-184442.log`    |    GPT1T_2L    |       8        | 2023-10-15-184442 | 2023-10-15-184727 | 184442 | 184727 |  285   |
> |    `foremans-login12-nhosts2-ngpu8-2023-10-15-185952.log`    |    GPT1T_1L    |       8        | 2023-10-15-185952 | 2023-10-15-190046 | 185952 | 190046 |  4094  |
> |   `foremans-nid008344-nhosts2-ngpu8-2023-10-15-191508.log`   |    GPT2_7B     |       8        | 2023-10-15-191508 | 2023-10-15-191608 | 191508 | 191608 |  100   |
> |   `foremans-nid008344-nhosts2-ngpu8-2023-10-15-192404.log`   |    GPT2_7B     |       8        | 2023-10-15-192404 | 2023-10-15-192504 | 192404 | 192504 |  100   |
> |   `foremans-nid008344-nhosts2-ngpu8-2023-10-15-193041.log`   |    GPT2_7B     |       8        | 2023-10-15-193041 | 2023-10-15-193137 | 193041 | 193137 |   96   |
> |   `foremans-nid008344-nhosts2-ngpu8-2023-10-15-193448.log`   |    GPT2_7B     |       8        | 2023-10-15-193448 | 2023-10-15-193540 | 193448 | 193540 |   92   |
> |   `foremans-login12-nhosts4-ngpu16-2023-10-15-195802.log`    |    GPT1T_1L    |       16       | 2023-10-15-195802 | 2023-10-15-195904 | 195802 | 195904 |  102   |
> |   `foremans-login12-nhosts4-ngpu16-2023-10-15-200019.log`    |    GPT2_7B     |       16       | 2023-10-15-200019 | 2023-10-15-200258 | 200019 | 200258 |  239   |
> |   `foremans-login12-nhosts4-ngpu16-2023-10-15-200902.log`    |    GPT2_7B     |       16       | 2023-10-15-200902 | 2023-10-15-201239 | 200902 | 201239 |  337   |
> |   `foremans-login12-nhosts4-ngpu16-2023-10-15-201524.log`    |    GPT2_7B     |       16       | 2023-10-15-201524 | 2023-10-15-201612 | 201524 | 201612 |   88   |
> |   `foremans-login12-nhosts4-ngpu16-2023-10-15-201834.log`    |    GPT2_7B     |       16       | 2023-10-15-201834 | 2023-10-15-201923 | 201834 | 201923 |   89   |
> |   `foremans-login12-nhosts4-ngpu16-2023-10-15-202402.log`    |    GPT2_7B     |       16       | 2023-10-15-202402 | 2023-10-15-202501 | 202402 | 202501 |   99   |
> |   `foremans-login12-nhosts4-ngpu16-2023-10-15-202606.log`    |    GPT2_7B     |       16       | 2023-10-15-202606 | 2023-10-15-202713 | 202606 | 202713 |  107   |
> |   `foremans-nid008344-nhosts2-ngpu8-2023-10-16-084033.log`   |    GPT1T_1L    |       8        | 2023-10-16-084033 | 2023-10-16-084212 | 84033  | 84212  |  179   |
> |   `foremans-nid008344-nhosts2-ngpu8-2023-10-16-084628.log`   |    GPT1T_1L    |       8        | 2023-10-16-084628 | 2023-10-16-084728 | 84628  | 84728  |  100   |
> |   `foremans-nid008344-nhosts2-ngpu8-2023-10-16-085401.log`   |    GPT1T_1L    |       8        | 2023-10-16-085401 | 2023-10-16-085505 | 85401  | 85505  |  104   |
> |   `foremans-nid008344-nhosts2-ngpu8-2023-10-16-090142.log`   |    GPT1T_1L    |       8        | 2023-10-16-090142 | 2023-10-16-090305 | 90142  | 90305  |  163   |
> |   `foremans-nid008344-nhosts2-ngpu8-2023-10-16-093404.log`   | actCkpt_GPT13B |       8        | 2023-10-16-093404 | 2023-10-16-093504 | 93404  | 93504  |  100   |
> |  `foremans-nid008572-nhosts4-ngpu16-2023-10-16-101437.log`   |    GPT1T_1L    |       16       | 2023-10-16-101437 | 2023-10-16-101549 | 101437 | 101549 |  112   |
> |  `foremans-nid008396-nhosts4-ngpu16-2023-10-16-101512.log`   |    GPT1T_1L    |       16       | 2023-10-16-101512 | 2023-10-16-101615 | 101512 | 101615 |  103   |
> |  `foremans-nid008396-nhosts4-ngpu16-2023-10-16-102217.log`   | actCkpt_GPT25B |       16       | 2023-10-16-102217 | 2023-10-16-102452 | 102217 | 102452 |  235   |
> |  `foremans-nid008396-nhosts4-ngpu16-2023-10-16-102750.log`   | actCkpt_GPT25B |       16       | 2023-10-16-102750 | 2023-10-16-103243 | 102750 | 103243 |  493   |
> |  `foremans-nid008572-nhosts4-ngpu16-2023-10-16-103113.log`   | actCkpt_GPT25B |       16       | 2023-10-16-103113 | 2023-10-16-103237 | 103113 | 103237 |  124   |
> |  `foremans-nid008396-nhosts4-ngpu16-2023-10-16-104037.log`   | actCkpt_GPT25B |       16       | 2023-10-16-104037 | 2023-10-16-104148 | 104037 | 104148 |  111   |
> |  `foremans-nid008396-nhosts4-ngpu16-2023-10-16-104819.log`   | actCkpt_GPT25B |       16       | 2023-10-16-104819 | 2023-10-16-110002 | 104819 | 110002 |  5183  |
> |  `foremans-nid008396-nhosts4-ngpu16-2023-10-16-110119.log`   | actCkpt_GPT25B |       16       | 2023-10-16-110119 | 2023-10-16-110225 | 110119 | 110225 |  106   |
> |  `foremans-nid008701-nhosts4-ngpu16-2023-10-16-113715.log`   | actCkpt_GPT25B |       16       | 2023-10-16-113715 | 2023-10-16-113824 | 113715 | 113824 |  109   |
> |  `foremans-nid008701-nhosts4-ngpu16-2023-10-16-114236.log`   |    GPT1T_1L    |       16       | 2023-10-16-114236 | 2023-10-16-114338 | 114236 | 114338 |  102   |
> |  `foremans-nid008701-nhosts4-ngpu16-2023-10-16-114610.log`   |    GPT1T_1L    |       16       | 2023-10-16-114610 | 2023-10-16-114711 | 114610 | 114711 |  101   |
> |  `foremans-nid008701-nhosts4-ngpu16-2023-10-16-114819.log`   |    GPT1T_2L    |       16       | 2023-10-16-114819 | 2023-10-16-114953 | 114819 | 114953 |  134   |
> |  `foremans-nid008701-nhosts4-ngpu16-2023-10-16-131058.log`   |    GPT1T_2L    |       16       | 2023-10-16-131058 | 2023-10-16-131203 | 131058 | 131203 |  145   |
> |   `foremans-nid008576-nhosts1-ngpu4-2023-10-16-151427.log`   |    GPT1T_1L    |       4        | 2023-10-16-151427 | 2023-10-16-151600 | 151427 | 151600 |  173   |
> |   `foremans-nid008576-nhosts1-ngpu4-2023-10-16-152528.log`   |    GPT1T_1L    |       4        | 2023-10-16-152528 | 2023-10-16-152640 | 152528 | 152640 |  112   |
> |   `foremans-nid008224-nhosts1-ngpu4-2023-10-16-175717.log`   |    GPT1T_1L    |       4        | 2023-10-16-175717 | 2023-10-16-175829 | 175717 | 175829 |  112   |
> |   `foremans-nid008224-nhosts1-ngpu4-2023-10-16-180457.log`   |    GPT1T_1L    |       4        | 2023-10-16-180457 | 2023-10-16-180605 | 180457 | 180605 |  148   |
> |   `foremans-nid008224-nhosts1-ngpu4-2023-10-16-183116.log`   |    GPT1T_1L    |       4        | 2023-10-16-183116 | 2023-10-16-183216 | 183116 | 183216 |  100   |
> |   `foremans-nid008224-nhosts1-ngpu4-2023-10-16-183921.log`   |    GPT1T_1L    |       4        | 2023-10-16-183921 | 2023-10-16-184033 | 183921 | 184033 |  112   |
> |   `foremans-nid008237-nhosts1-ngpu4-2023-10-16-215614.log`   |    GPT1T_1L    |       4        | 2023-10-16-215614 | 2023-10-16-215815 | 215614 | 215815 |  201   |
> |   `foremans-nid008385-nhosts1-ngpu4-2023-10-17-052944.log`   |    GPT1T_1L    |       4        | 2023-10-17-052944 | 2023-10-17-053139 | 52944  | 53139  |  195   |
> |   `foremans-nid008385-nhosts1-ngpu4-2023-10-17-053529.log`   |    GPT1T_1L    |       4        | 2023-10-17-053529 | 2023-10-17-053650 | 53529  | 53650  |  121   |
> |   `foremans-nid008385-nhosts1-ngpu4-2023-10-17-053910.log`   |    GPT1T_1L    |       4        | 2023-10-17-053910 | 2023-10-17-054120 | 53910  | 54120  |  210   |
> |   `foremans-nid008385-nhosts1-ngpu4-2023-10-17-054238.log`   |    GPT2_7B     |       4        | 2023-10-17-054238 | 2023-10-17-054346 | 54238  | 54346  |  108   |
> |   `foremans-nid008385-nhosts1-ngpu4-2023-10-17-060418.log`   |    GPT1T_1L    |       4        | 2023-10-17-060418 | 2023-10-17-060600 | 60418  | 60600  |  182   |
> |   `foremans-nid008385-nhosts1-ngpu4-2023-10-17-061514.log`   |    GPT1T_1L    |       4        | 2023-10-17-061514 | 2023-10-17-061653 | 61514  | 61653  |  139   |
> |   `foremans-nid008385-nhosts1-ngpu4-2023-10-17-062102.log`   |    GPT1T_1L    |       4        | 2023-10-17-062102 | 2023-10-17-062252 | 62102  | 62252  |  150   |
> |   `foremans-nid008385-nhosts1-ngpu4-2023-10-17-062445.log`   |    GPT1T_1L    |       4        | 2023-10-17-062445 | 2023-10-17-062720 | 62445  | 62720  |  275   |
> |   `foremans-nid008333-nhosts2-ngpu8-2023-10-17-064643.log`   |    GPT1T_1L    |       8        | 2023-10-17-064643 | 2023-10-17-064848 | 64643  | 64848  |  205   |
> |   `foremans-nid008333-nhosts2-ngpu8-2023-10-17-065806.log`   |    GPT1T_2L    |       8        | 2023-10-17-065806 | 2023-10-17-070003 | 65806  | 70003  |  4197  |
> |   `foremans-nid008333-nhosts2-ngpu8-2023-10-17-075152.log`   |    GPT1T_2L    |       8        | 2023-10-17-075152 | 2023-10-17-075502 | 75152  | 75502  |  350   |
> |   `foremans-nid008333-nhosts2-ngpu8-2023-10-17-080059.log`   |    GPT1T_2L    |       8        | 2023-10-17-080059 | 2023-10-17-080434 | 80059  | 80434  |  375   |
> |   `foremans-nid008333-nhosts2-ngpu8-2023-10-17-081404.log`   |    GPT1T_2L    |       8        | 2023-10-17-081404 | 2023-10-17-081920 | 81404  | 81920  |  516   |
> |   `foremans-nid008228-nhosts1-ngpu4-2023-10-17-090344.log`   |    GPT1T_1L    |       4        | 2023-10-17-090344 | 2023-10-17-090714 | 90344  | 90714  |  370   |
> |   `foremans-nid008228-nhosts1-ngpu4-2023-10-17-100759.log`   |    GPT1T_1L    |       4        | 2023-10-17-100759 | 2023-10-17-100957 | 100759 | 100957 |  198   |
> |  `foremans-nid008404-nhosts4-ngpu16-2023-10-17-182501.log`   |    GPT1T_1L    |       16       | 2023-10-17-182501 | 2023-10-17-184001 | 182501 | 184001 |  1500  |
> |  `foremans-nid008404-nhosts4-ngpu16-2023-10-17-193736.log`   |    GPT1T_1L    |       16       | 2023-10-17-193736 | 2023-10-17-193856 | 193736 | 193856 |  120   |
> |  `foremans-nid008404-nhosts4-ngpu16-2023-10-17-195432.log`   |    GPT1T_1L    |       16       | 2023-10-17-195432 | 2023-10-17-195536 | 195432 | 195536 |  104   |
> |  `foremans-nid008404-nhosts4-ngpu16-2023-10-17-201659.log`   |    GPT1T_2L    |       16       | 2023-10-17-201659 | 2023-10-17-201823 | 201659 | 201823 |  164   |
> |  `foremans-nid008404-nhosts4-ngpu16-2023-10-17-202949.log`   |    GPT1T_2L    |       16       | 2023-10-17-202949 | 2023-10-17-203054 | 202949 | 203054 |  105   |
> |  `foremans-nid008404-nhosts4-ngpu16-2023-10-17-205848.log`   |    GPT1T_1L    |       16       | 2023-10-17-205848 | 2023-10-17-205952 | 205848 | 205952 |  104   |
> |  `foremans-nid008577-nhosts8-ngpu32-2023-10-17-213244.log`   |    GPT1T_1L    |       32       | 2023-10-17-213244 | 2023-10-17-213406 | 213244 | 213406 |  162   |
> |  `foremans-nid008577-nhosts8-ngpu32-2023-10-17-213558.log`   |    GPT1T_1L    |       32       | 2023-10-17-213558 | 2023-10-17-213720 | 213558 | 213720 |  162   |
> |  `foremans-nid008577-nhosts8-ngpu32-2023-10-17-214900.log`   |    GPT1T_2L    |       32       | 2023-10-17-214900 | 2023-10-17-214959 | 214900 | 214959 |   59   |
> |  `foremans-nid008577-nhosts8-ngpu32-2023-10-17-215201.log`   |    GPT1T_2L    |       32       | 2023-10-17-215201 | 2023-10-17-215309 | 215201 | 215309 |  108   |
> |  `foremans-nid008577-nhosts8-ngpu32-2023-10-17-215612.log`   |    GPT1T_2L    |       32       | 2023-10-17-215612 | 2023-10-17-215726 | 215612 | 215726 |  114   |
> |  `foremans-nid008577-nhosts8-ngpu32-2023-10-17-215938.log`   |    GPT1T_2L    |       32       | 2023-10-17-215938 | 2023-10-17-220044 | 215938 | 220044 |  4106  |
> |  `foremans-nid008529-nhosts8-ngpu32-2023-10-18-110001.log`   |    GPT1T_4L    |       32       | 2023-10-18-110001 | 2023-10-18-110143 | 110001 | 110143 |  142   |
> |  `foremans-nid008529-nhosts8-ngpu32-2023-10-18-110424.log`   |    GPT1T_8L    |       32       | 2023-10-18-110424 | 2023-10-18-110550 | 110424 | 110550 |  126   |
> |  `foremans-nid008244-nhosts4-ngpu16-2023-10-18-110821.log`   |    GPT1T_8L    |       16       | 2023-10-18-110821 | 2023-10-18-110952 | 110821 | 110952 |  131   |
> |  `foremans-nid008529-nhosts8-ngpu32-2023-10-18-111345.log`   |    GPT1T_8L    |       32       | 2023-10-18-111345 | 2023-10-18-111458 | 111345 | 111458 |  113   |
> |  `foremans-nid008197-nhosts16-ngpu64-2023-10-18-112531.log`  |   GPT1T_16L    |       64       | 2023-10-18-112531 | 2023-10-18-112728 | 112531 | 112728 |  197   |
> |  `foremans-nid008456-nhosts16-ngpu64-2023-10-18-113119.log`  |   GPT1T_16L    |       64       | 2023-10-18-113119 | 2023-10-18-113343 | 113119 | 113343 |  224   |
> |  `foremans-nid008244-nhosts4-ngpu16-2023-10-18-113131.log`   |    GPT1T_4L    |       16       | 2023-10-18-113131 | 2023-10-18-113257 | 113131 | 113257 |  126   |
> |  `foremans-nid008244-nhosts4-ngpu16-2023-10-18-113920.log`   |    GPT1T_4L    |       16       | 2023-10-18-113920 | 2023-10-18-114157 | 113920 | 114157 |  237   |
> |  `foremans-nid008197-nhosts16-ngpu64-2023-10-18-114549.log`  |   GPT1T_16L    |       64       | 2023-10-18-114549 | 2023-10-18-114721 | 114549 | 114721 |  172   |
> |  `foremans-nid008456-nhosts16-ngpu64-2023-10-18-114636.log`  |   GPT1T_16L    |       64       | 2023-10-18-114636 | 2023-10-18-114805 | 114636 | 114805 |  169   |
> |  `foremans-nid008244-nhosts4-ngpu16-2023-10-18-115808.log`   |    GPT1T_4L    |       16       | 2023-10-18-115808 | 2023-10-18-120146 | 115808 | 120146 |  4338  |
> |  `foremans-nid008456-nhosts16-ngpu64-2023-10-18-123039.log`  |   GPT1T_16L    |       64       | 2023-10-18-123039 | 2023-10-18-123221 | 123039 | 123221 |  182   |
> |   `foremans-nid008389-nhosts2-ngpu8-2023-10-18-123135.log`   |    GPT1T_4L    |       8        | 2023-10-18-123135 | 2023-10-18-123300 | 123135 | 123300 |  165   |
> |  `foremans-nid008244-nhosts4-ngpu16-2023-10-18-123206.log`   |    GPT1T_4L    |       16       | 2023-10-18-123206 | 2023-10-18-123352 | 123206 | 123352 |  146   |
> |  `foremans-nid008456-nhosts16-ngpu64-2023-10-18-125022.log`  |   GPT1T_16L    |       64       | 2023-10-18-125022 | 2023-10-18-125146 | 125022 | 125146 |  124   |
> |  `foremans-nid008256-nhosts8-ngpu32-2023-10-22-122736.log`   |    GPT1T_8L    |       32       | 2023-10-22-122736 | 2023-10-22-122844 | 122736 | 122844 |  108   |
> |  `foremans-nid008256-nhosts8-ngpu32-2023-10-22-123824.log`   |    GPT1T_8L    |       32       | 2023-10-22-123824 | 2023-10-22-123945 | 123824 | 123945 |  121   |
> |  `foremans-nid008256-nhosts8-ngpu32-2023-10-22-130148.log`   |    GPT1T_8L    |       32       | 2023-10-22-130148 | 2023-10-22-130256 | 130148 | 130256 |  108   |
> |  `foremans-nid008256-nhosts8-ngpu32-2023-10-22-131746.log`   |    GPT1T_8L    |       32       | 2023-10-22-131746 | 2023-10-22-131909 | 131746 | 131909 |  163   |
> |  `foremans-nid008256-nhosts8-ngpu32-2023-10-22-132700.log`   |    GPT1T_8L    |       32       | 2023-10-22-132700 | 2023-10-22-132817 | 132700 | 132817 |  117   |
> |  `foremans-nid008256-nhosts8-ngpu32-2023-10-22-133459.log`   |    GPT1T_8L    |       32       | 2023-10-22-133459 | 2023-10-22-133708 | 133459 | 133708 |  249   |
> |  `foremans-nid008380-nhosts4-ngpu16-2023-10-22-175049.log`   | actCkpt_GPT25B |       16       | 2023-10-22-175049 | 2023-10-22-175230 | 175049 | 175230 |  181   |
> |  `foremans-nid008649-nhosts4-ngpu16-2023-10-22-192352.log`   |    GPT1T_4L    |       16       | 2023-10-22-192352 | 2023-10-22-192530 | 192352 | 192530 |  178   |
> |  `foremans-nid008212-nhosts16-ngpu64-2023-10-23-081527.log`  |    GPT1T_8L    |       64       | 2023-10-23-081527 | 2023-10-23-081702 | 81527  | 81702  |  175   |
> |   `foremans-nid008344-nhosts2-ngpu8-2023-10-23-091436.log`   |    GPT1T_2L    |       8        | 2023-10-23-091436 | 2023-10-23-091610 | 91436  | 91610  |  174   |
> | `foremans-nid008197-nhosts32-ngpu128-2023-10-24-102617.log`  |   GPT1T_32L    |      128       | 2023-10-24-102617 | 2023-10-24-102826 | 102617 | 102826 |  209   |
> | `foremans-nid008192-nhosts64-ngpu256-2023-10-24-191748.log`  |   GPT1T_64L    |      256       | 2023-10-24-191748 | 2023-10-24-192021 | 191748 | 192021 |  273   |
> | `foremans-nid008192-nhosts128-ngpu512-2023-10-24-201243.log` |   GPT1T_128L   |      512       | 2023-10-24-201243 | 2023-10-24-201629 | 201243 | 201629 |  386   |
> | `foremans-nid008192-nhosts128-ngpu512-2023-10-26-005401.log` |   GPT1T_128L   |      512       | 2023-10-26-005401 | 2023-10-26-005811 |  5401  |  5811  |  410   |
> | `foremans-nid008192-nhosts32-ngpu128-2023-10-26-082710.log`  |   GPT1T_32L    |      128       | 2023-10-26-082710 | 2023-10-26-083049 | 82710  | 83049  |  339   |
> |   `foremans-nid008585-nhosts2-ngpu8-2023-10-31-044203.log`   |    GPT1T_2L    |       8        | 2023-10-31-044203 | 2023-10-31-044533 | 44203  | 44533  |  330   |
> |  `foremans-nid008272-nhosts4-ngpu16-2023-10-31-072717.log`   |    GPT1T_4L    |       16       | 2023-10-31-072717 | 2023-10-31-073131 | 72717  | 73131  |  414   |
> |  `foremans-nid008221-nhosts8-ngpu32-2023-10-31-083055.log`   |    GPT1T_8L    |       32       | 2023-10-31-083055 | 2023-10-31-083545 | 83055  | 83545  |  490   |
> |  `foremans-nid008196-nhosts16-ngpu64-2023-10-31-100336.log`  |   GPT1T_16L    |       64       | 2023-10-31-100336 | 2023-10-31-100848 | 100336 | 100848 |  512   |
> |   `foremans-nid008285-nhosts2-ngpu8-2023-11-01-200430.log`   |    GPT1T_2L    |       8        | 2023-11-01-200430 | 2023-11-01-200829 | 200430 | 200829 |  399   |
> |  `foremans-nid008193-nhosts8-ngpu32-2023-11-01-201702.log`   |    GPT1T_8L    |       32       | 2023-11-01-201702 | 2023-11-01-202131 | 201702 | 202131 |  429   |
> |  `foremans-nid008240-nhosts16-ngpu64-2023-11-01-210454.log`  |   GPT1T_16L    |       64       | 2023-11-01-210454 | 2023-11-01-211007 | 210454 | 211007 |  553   |
> |   `foremans-nid008321-nhosts2-ngpu8-2023-11-02-154438.log`   |    GPT1T_2L    |       8        | 2023-11-02-154438 | 2023-11-02-154949 | 154438 | 154949 |  511   |
> | `foremans-nid008192-nhosts128-ngpu512-2023-11-04-001717.log` |   GPT1T_128L   |      512       | 2023-11-04-001717 | 2023-11-04-002124 |  1717  |  2124  |  407   |

## Minimal Working Example

- As for 3:

  > If we need to report the startup time for the DL applications, do we
  > need to collect measurements using the actual Aurora NRE workloads
  > or some small benchmarking test cases? For example, we can try to
  > recreate the typical start-up scenarios, like library imports, and
  > measure those separately as shown below.

  - I’ve been working on a library to help simplify this:

    [ `ezpz`](https://github.com/saforem2/ezpz)  
    <span class="dim-text">*Minimal library that handles the
    initialization of distributed training*</span>

- [x]   Working on Aurora, example:

  - Setup / Install:

    ``` bash
    # launch job
    $ qsub -q EarlyAppAccess -A Aurora_Deployment -l walltime=2:00:00 -l select=4 -I

    # load frameworks
    $ module use -a /soft/modulefiles ; module --ignore_cache load frameworks
    $ module load frameworks/.2023.12.15.001

    # install `ezpz`
    $ git clone https://github.com/saforem2/ezpz
    $ cd ezpz
    $ mkdir -p venvs/aurora/2023.12.15.001
    $ python3 -m venv venvs/aurora/2023.12.15.001 --system-site-packages
    $ source venvs/aurora/2023.12.15.001/bin/activate
    $ python3 -m pip install -e .

    # print job info and define `launch` alias
    $ source ezpz/src/ezpz/bin/savejobenv
    ┌──────────────────────────────────────────────────────────────────
    │ [Hosts]:
    │     • x4415c6s5b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov
    x4415c6s6b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov
    x4415c6s7b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov
    x4415c7s0b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov
    └──────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [DIST INFO]:
    │     • Loading job env from: /home/foremans/.pbsenv
    │     • HOSTFILE: /var/spool/pbs/aux/297306.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    │     • NHOSTS: 4
    │     • NGPU_PER_HOST: 12
    │     • NGPUS (NHOSTS x NGPU_PER_HOST): 48
    │     • DIST_LAUNCH: mpiexec --verbose --envall -n 48 -ppn 12 --hostfile /var/spool/pbs/aux/297306.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    │     • Defining alias: launch: aliased to mpiexec --verbose --envall -n 48 -ppn 12 --hostfile /var/spool/pbs/aux/297306.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    └──────────────────────────────────────────────────────────────────
    ```

  - Launch with `framework=pytorch`, `backend=DDP`:

    ``` bash
    # ----------------------------------------------------------
    # launch + startup on all workers with
    # • `framework` ∈ {`pytorch`, `tensorflow`}
    # • `backend` ∈ {`horovod`, `deepspeed`, `DDP`}
    # where `deepspeed` and `DDP` only available for `pytorch`
    # ----------------------------------------------------------
    $ launch python3 -m ezpz framework=pytorch backend=DDP
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:24][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:25][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:26][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:26][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:26][INFO][dist.py:243] - Using DDP for distributed training
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:26][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:27][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:28][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:28][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:29][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:29][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:29][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:30][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:30][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:30][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:30][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:30][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:30][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:34][INFO][dist.py:292] - Using device='xpu'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][WARNING][dist.py:104] - Using backend='ccl'
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 1 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 2 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 3 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 4 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 0 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 5 / 47
    [2023-12-19 13:33:35][INFO][__main__.py:49] - {
        "_target_": "ezpz.configs.TrainConfig",
        "framework": "pytorch",
        "backend": "DDP",
        "ds_config_path": null,
        "port": null,
        "seed": null,
        "use_wandb": true,
        "wandb_project_name": null,
        "precision": null,
        "ngpus": null
    }
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 9 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 10 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 11 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 7 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 8 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 6 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 12 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 13 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 14 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 15 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 18 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 19 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 20 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 21 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 22 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 23 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 24 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 25 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 26 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 27 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 30 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 16 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 17 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 28 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 32 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 33 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 36 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 37 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 38 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 39 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 43 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 46 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 29 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 47 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 31 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 34 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 35 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 42 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 41 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 44 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 45 / 47
    [2023-12-19 13:33:35][INFO][dist.py:307] - RANK: 40 / 47
    [2023-12-19 13:33:47][INFO][dist.py:415] - Setting up wandb from rank: 0
    [2023-12-19 13:33:47][INFO][dist.py:416] - Using: WB PROJECT: ezpz
    [2023-12-19 13:33:58][INFO][dist.py:448] - W&B RUN: [flowing-wood-8](https://wandb.ai/l2hmc-qcd/ezpz/runs/uya29gm5)
    [2023-12-19 13:33:58][INFO][dist.py:490] - Running on x4415c6s5b0n0.hostmgmt2415.cm.aurora.alcf.anl.gov
    [2023-12-19 13:33:58][INFO][dist.py:506] - Reading hosts from /var/spool/pbs/aux/297306.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    [2023-12-19 13:33:58][INFO][__main__.py:57] - Output dir: /lus/gecko/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/src/ezpz/outputs/runs/pytorch/DDP/2023-12-19/13-33-17
    [2023-12-19 13:33:58][CRITICAL][dist.py:519] - 🚀 flowing-wood-8
    [2023-12-19 13:33:58][CRITICAL][dist.py:520] - 🔗 https://wandb.ai/l2hmc-qcd/ezpz/runs/uya29gm5
    [2023-12-19 13:33:58][CRITICAL][dist.py:521] - 📂/: /lus/gecko/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/src/ezpz/outputs/runs/pytorch/DDP/2023-12-19/13-33-17/wandb/run-20231219_133354-uya29gm5/files
    [2023-12-19 13:33:58][INFO][dist.py:563] - Adding /lus/gecko/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/src/ezpz/ezpz-pt-DDP-xpu.log to W&B artifact...
    [2023-12-19 13:33:58][INFO][dist.py:563] - Adding /lus/gecko/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/src/ezpz/outputs/runs/pytorch/DDP/2023-12-19/13-33-17/__main__.log to W&B artifact...
    [2023-12-19 13:33:58][INFO][dist.py:563] - Adding /lus/gecko/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/src/ezpz/outputs/runs/pytorch/DDP/2023-12-19/13-33-17/main_debug.log to W&B artifact...
    [2023-12-19 13:33:58][INFO][dist.py:563] - Adding /lus/gecko/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/src/ezpz/outputs/runs/pytorch/DDP/2023-12-19/13-33-16/__main__.log to W&B artifact...
    ```
