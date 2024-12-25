## 📝 Example

1.  Clone repo:

    ``` bash
    git clone https://github.com/saforem2/ezpz
    cd ezpz
    ```

2.  Setup environment:

    ``` bash
    export PBS_O_WORKDIR=$(pwd) && source src/ezpz/bin/utils.sh && ezpz_setup_env
    ```

    <details closed><summary>Output</summary>

    ``` bash
    $ export PBS_O_WORKDIR=$(pwd) && source src/ezpz/bin/utils.sh && ezpz_setup_env
    Using WORKING_DIR: /gila/Aurora_deployment/foremans/projects/saforem2/ezpz
    No conda_prefix OR virtual_env found in environment...
    Setting up conda...

    Due to MODULEPATH changes, the following have been reloaded:
      1) mpich/icc-all-pmix-gpu/20240717

    The following have been reloaded with a version change:
      1) oneapi/eng-compiler/2024.07.30.002 => oneapi/release/2024.2.1

    Found conda at: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1
    No VIRTUAL_ENV found in environment!
        - Trying to setup from /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1
        - Using VENV_DIR=/gila/Aurora_deployment/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2024.2.1_u1

        - Creating a new virtual env on top of aurora_nre_models_frameworks-2024.2.1_u1 in /gila/Aurora_deployment/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2024.2.1_u1
    [python] Using /gila/Aurora_deployment/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3

    [🍋 ezpz/bin/utils.sh]
        • USER=foremans
        • MACHINE=sunspot
        • HOST=x1922c5s0b0n0
        • TSTAMP=2024-11-28-133756

    [ezpz_setup_host_pbs]
        • Using hostfile: /var/spool/pbs/aux/10283088.amn-0001
        • Found in environment:
            • HOSTFILE: /var/spool/pbs/aux/10283088.amn-0001
            • Writing PBS vars to: /home/foremans/.pbsenv

    [ezpz_save_pbs_env]
        • Setting:
            • HOSTFILE: /var/spool/pbs/aux/10283088.amn-0001
            • JOBENV_FILE: /home/foremans/.pbsenv

    [HOSTS]
        • [host:0] - x1922c5s0b0n0.hostmgmt2001.cm.americas.sgi.com
        • [host:1] - x1922c5s2b0n0.hostmgmt2001.cm.americas.sgi.com
        • [host:2] - x1922c5s4b0n0.hostmgmt2001.cm.americas.sgi.com
        • [host:3] - x1922c6s1b0n0.hostmgmt2001.cm.americas.sgi.com

    [DIST INFO]
        • NGPUS=48
        • NHOSTS=4
        • NGPU_PER_HOST=12
        • HOSTFILE=/var/spool/pbs/aux/10283088.amn-0001
        • DIST_LAUNCH=mpiexec --verbose --envall -n 48 -ppn 12 --hostfile /var/spool/pbs/aux/10283088.amn-0001 --cpu-bind depth -d 8

    [LAUNCH]:
        • To launch across all available GPUs, use: launch

          launch = mpiexec --verbose --envall -n 48 -ppn 12 --hostfile /var/spool/pbs/aux/10283088.amn-0001 --cpu-bind depth -d 8

    took: 0h:00m:12s
    ```

    </details>

3.  Install `ezpz`:

    ``` bash
    python3 -m pip install -e "." --require-virtualenv
    ```

4.  Check `launch`:

    ``` bash
    $ which launch
    launch: aliased to mpiexec --verbose --envall -n 48 -ppn 12 --hostfile /var/spool/pbs/aux/10283088.amn-0001 --cpu-bind depth -d 8
    ```

5.  Run `ezpz.test_dist`:

    ``` bash
    launch python3 src/ezpz/test_dist.py
    ```

    <details closed><summary>Output</summary>

    ``` bash
    #[🐍 aurora_nre_models_frameworks-2023.2.1_u1](👻 aurora_nre_models_frameworks-2024.2.1_u1)
    #[🌻][01:42:37 PM][foremans@x1922c5s0b0n0][…/ezpz][🌱 saforem2-patch-1]via ⨁ v1.4.552
    $ launch python3 -m ezpz.test_dist
    Disabling local launch: multi-node application
    Connected to tcp://x1922c5s0b0n0.hostmgmt2001.cm.americas.sgi.com:7919
    Found executable /gila/Aurora_deployment/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
    Launching application ecd9868b-2b3b-4a0e-b2e6-79769ecc7eff
    [2024-11-28 13:43:41.385960][INFO][dist.py:348] - [device='xpu'][rank=3/47][local_rank=3/11][node=3/3]
    [2024-11-28 13:43:41.386603][INFO][dist.py:348] - [device='xpu'][rank=1/47][local_rank=1/11][node=1/3]
    [2024-11-28 13:43:41.386605][INFO][dist.py:348] - [device='xpu'][rank=2/47][local_rank=2/11][node=2/3]
    [2024-11-28 13:43:41.386834][INFO][dist.py:348] - [device='xpu'][rank=6/47][local_rank=6/11][node=2/3]
    [2024-11-28 13:43:41.387707][INFO][dist.py:348] - [device='xpu'][rank=8/47][local_rank=8/11][node=0/3]
    [2024-11-28 13:43:41.387290][INFO][dist.py:348] - [device='xpu'][rank=4/47][local_rank=4/11][node=0/3]
    [2024-11-28 13:43:41.387235][INFO][dist.py:348] - [device='xpu'][rank=10/47][local_rank=10/11][node=2/3]
    [2024-11-28 13:43:41.387362][INFO][dist.py:348] - [device='xpu'][rank=11/47][local_rank=11/11][node=3/3]
    [2024-11-28 13:43:41.387761][INFO][dist.py:348] - [device='xpu'][rank=5/47][local_rank=5/11][node=1/3]
    [2024-11-28 13:43:41.387958][INFO][dist.py:348] - [device='xpu'][rank=9/47][local_rank=9/11][node=1/3]
    [2024-11-28 13:43:46.384505][INFO][dist.py:348] - [device='xpu'][rank=7/47][local_rank=7/11][node=3/3]
    [2024-11-28 13:44:32.816252][INFO][dist.py:348] - [device='xpu'][rank=15/47][local_rank=3/11][node=3/3]
    [2024-11-28 13:44:32.821175][INFO][dist.py:348] - [device='xpu'][rank=17/47][local_rank=5/11][node=1/3]
    [2024-11-28 13:44:32.824021][INFO][dist.py:348] - [device='xpu'][rank=22/47][local_rank=10/11][node=2/3]
    [2024-11-28 13:44:32.825905][INFO][dist.py:348] - [device='xpu'][rank=41/47][local_rank=5/11][node=1/3]
    [2024-11-28 13:44:32.826590][INFO][dist.py:348] - [device='xpu'][rank=38/47][local_rank=2/11][node=2/3]
    [2024-11-28 13:44:32.838048][INFO][dist.py:348] - [device='xpu'][rank=23/47][local_rank=11/11][node=3/3]
    [2024-11-28 13:44:32.838526][INFO][dist.py:348] - [device='xpu'][rank=21/47][local_rank=9/11][node=1/3]
    [2024-11-28 13:44:32.838825][INFO][dist.py:348] - [device='xpu'][rank=19/47][local_rank=7/11][node=3/3]
    [2024-11-28 13:44:32.838817][INFO][dist.py:348] - [device='xpu'][rank=36/47][local_rank=0/11][node=0/3]
    [2024-11-28 13:44:32.838665][INFO][dist.py:348] - [device='xpu'][rank=35/47][local_rank=11/11][node=3/3]
    [2024-11-28 13:44:32.839033][INFO][dist.py:348] - [device='xpu'][rank=25/47][local_rank=1/11][node=1/3]
    [2024-11-28 13:44:32.838855][INFO][dist.py:348] - [device='xpu'][rank=26/47][local_rank=2/11][node=2/3]
    [2024-11-28 13:44:32.839144][INFO][dist.py:348] - [device='xpu'][rank=33/47][local_rank=9/11][node=1/3]
    [2024-11-28 13:44:32.840785][INFO][dist.py:348] - [device='xpu'][rank=37/47][local_rank=1/11][node=1/3]
    [2024-11-28 13:44:32.840740][INFO][dist.py:348] - [device='xpu'][rank=47/47][local_rank=11/11][node=3/3]
    [2024-11-28 13:44:32.844721][INFO][dist.py:348] - [device='xpu'][rank=46/47][local_rank=10/11][node=2/3]
    [2024-11-28 13:44:32.845202][INFO][dist.py:348] - [device='xpu'][rank=30/47][local_rank=6/11][node=2/3]
    [2024-11-28 13:44:32.845888][INFO][dist.py:348] - [device='xpu'][rank=18/47][local_rank=6/11][node=2/3]
    [2024-11-28 13:44:32.849905][INFO][dist.py:348] - [device='xpu'][rank=20/47][local_rank=8/11][node=0/3]
    [2024-11-28 13:44:32.849947][INFO][dist.py:348] - [device='xpu'][rank=32/47][local_rank=8/11][node=0/3]
    [2024-11-28 13:44:32.850232][INFO][dist.py:348] - [device='xpu'][rank=39/47][local_rank=3/11][node=3/3]
    [2024-11-28 13:44:32.850301][INFO][dist.py:348] - [device='xpu'][rank=44/47][local_rank=8/11][node=0/3]
    [2024-11-28 13:44:32.850795][INFO][dist.py:348] - [device='xpu'][rank=14/47][local_rank=2/11][node=2/3]
    [2024-11-28 13:44:32.851872][INFO][dist.py:348] - [device='xpu'][rank=28/47][local_rank=4/11][node=0/3]
    [2024-11-28 13:44:32.852078][INFO][dist.py:348] - [device='xpu'][rank=12/47][local_rank=0/11][node=0/3]
    [2024-11-28 13:44:32.853836][INFO][dist.py:348] - [device='xpu'][rank=16/47][local_rank=4/11][node=0/3]
    [2024-11-28 13:44:32.853997][INFO][dist.py:348] - [device='xpu'][rank=24/47][local_rank=0/11][node=0/3]
    [2024-11-28 13:44:32.855526][INFO][dist.py:348] - [device='xpu'][rank=13/47][local_rank=1/11][node=1/3]
    [2024-11-28 13:44:32.856609][INFO][dist.py:348] - [device='xpu'][rank=40/47][local_rank=4/11][node=0/3]
    [2024-11-28 13:44:32.857991][INFO][dist.py:348] - [device='xpu'][rank=42/47][local_rank=6/11][node=2/3]
    [2024-11-28 13:44:32.863239][INFO][dist.py:348] - [device='xpu'][rank=29/47][local_rank=5/11][node=1/3]
    [2024-11-28 13:44:32.864956][INFO][dist.py:348] - [device='xpu'][rank=45/47][local_rank=9/11][node=1/3]
    [2024-11-28 13:44:32.867388][INFO][dist.py:348] - [device='xpu'][rank=27/47][local_rank=3/11][node=3/3]
    [2024-11-28 13:44:32.867764][INFO][dist.py:348] - [device='xpu'][rank=43/47][local_rank=7/11][node=3/3]
    [2024-11-28 13:44:32.873106][INFO][dist.py:348] - [device='xpu'][rank=34/47][local_rank=10/11][node=2/3]
    [2024-11-28 13:44:32.877043][INFO][dist.py:348] - [device='xpu'][rank=31/47][local_rank=7/11][node=3/3]
    [2024-11-28 13:44:32.887609][INFO][dist.py:92] -

    [dist_info]:
      • DEVICE=xpu
      • DEVICE_ID=xpu:0
      • DISTRIBUTED_BACKEND=ccl
      • GPUS_PER_NODE=12
      • HOSTS=['x1922c5s0b0n0.hostmgmt2001.cm.americas.sgi.com', 'x1922c5s2b0n0.hostmgmt2001.cm.americas.sgi.com', 'x1922c5s4b0n0.hostmgmt2001.cm.americas.sgi.com', 'x1922c6s1b0n0.hostmgmt2001.cm.americas.sgi.com']
      • HOSTFILE=/var/spool/pbs/aux/10283088.amn-0001
      • HOSTNAME=x1922c5s0b0n0.hostmgmt2001.cm.americas.sgi.com
      • LOCAL_RANK=0
      • MACHINE=SunSpot
      • NUM_NODES=4
      • NGPUS=48
      • NGPUS_AVAILABLE=48
      • NODE_ID=0
      • RANK=0
      • SCHEDULER=PBS
      • WORLD_SIZE_TOTAL=48
      • WORLD_SIZE_IN_USE=48
      • LAUNCH_CMD=mpiexec --verbose --envall -n 48 -ppn 12 --hostfile /var/spool/pbs/aux/10283088.amn-0001 --cpu-bind depth -d 16


    [2024-11-28 13:44:32.933415][INFO][dist.py:725] - Using oneccl_bindings from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/__init__.py
    [2024-11-28 13:44:32.933861][INFO][dist.py:727] - Using ipex from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/intel_extension_for_pytorch/__init__.py
    [2024-11-28 13:44:32.934238][INFO][dist.py:728] - [0/48] Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
    [2024-11-28 13:44:32.940188][INFO][dist.py:348] - [device='xpu'][rank=0/47][local_rank=0/11][node=0/3]
    [2024-11-28 13:44:32.940746][WARNING][_logger.py:68] - Using [48 / 48] available "xpu" devices !!
    [2024-11-28 13:44:34.193788][INFO][dist.py:882] - Setting up wandb from rank: 0
    [2024-11-28 13:44:34.194312][INFO][dist.py:883] - Using: WB PROJECT: ezpz.test_dist
    wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
    wandb: Currently logged in as: foremans (aurora_gpt). Use `wandb login --relogin` to force relogin
    wandb: Tracking run with wandb version 0.18.7
    wandb: Run data is saved locally in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/wandb/run-20241128_134434-4mwyy84l
    wandb: Run `wandb offline` to turn off syncing.
    wandb: Syncing run driven-wind-683
    wandb: ⭐️ View project at https://wandb.ai/aurora_gpt/ezpz.test_dist
    wandb: 🚀 View run at https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/4mwyy84l
    [2024-11-28 13:44:35.481364][INFO][dist.py:908] - W&B RUN: [driven-wind-683](https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/4mwyy84l)
    [2024-11-28 13:44:35.495959][INFO][dist.py:304] - Updating wandb.run: driven-wind-683 config with "DIST_INFO"
    [2024-11-28 13:44:35.500222][INFO][dist.py:936] - Running on machine='SunSpot'
    [2024-11-28 13:44:35.501304][INFO][dist.py:92] -

    [CONFIG]:
      • warmup=0
      • log_freq=1
      • batch_size=64
      • input_size=128
      • output_size=128
      • dtype=torch.float32
      • device=xpu
      • world_size=48
      • train_iters=100


    [2024-11-28 13:44:35.575161][INFO][test_dist.py:147] - model=Network(
      (layers): Sequential(
        (0): Linear(in_features=128, out_features=1024, bias=True)
        (1): Linear(in_features=1024, out_features=512, bias=True)
        (2): Linear(in_features=512, out_features=256, bias=True)
        (3): Linear(in_features=256, out_features=128, bias=True)
        (4): Linear(in_features=128, out_features=128, bias=True)
      )
    )
    [2024-11-28 13:44:48.340596][INFO][test_dist.py:228] - iter=1 dt=0.019713 dtf=0.002959 dtb=0.016754 loss=1821.131958 sps=3246.551823
    [2024-11-28 13:44:48.346527][INFO][test_dist.py:228] - iter=2 dt=0.004083 dtf=0.000822 dtb=0.003261 loss=1348.032227 sps=15673.877930
    [2024-11-28 13:44:48.351387][INFO][test_dist.py:228] - iter=3 dt=0.003260 dtf=0.000735 dtb=0.002525 loss=1072.068359 sps=19634.479814
    [2024-11-28 13:44:48.356043][INFO][test_dist.py:228] - iter=4 dt=0.003117 dtf=0.000716 dtb=0.002401 loss=928.493774 sps=20534.414826
    [2024-11-28 13:44:48.360830][INFO][test_dist.py:228] - iter=5 dt=0.003181 dtf=0.000798 dtb=0.002384 loss=867.221558 sps=20116.568892
    [2024-11-28 13:44:48.365608][INFO][test_dist.py:228] - iter=6 dt=0.003146 dtf=0.000738 dtb=0.002408 loss=794.671204 sps=20341.929281
    [2024-11-28 13:44:48.370473][INFO][test_dist.py:228] - iter=7 dt=0.003149 dtf=0.000751 dtb=0.002398 loss=748.687500 sps=20324.551047
    [2024-11-28 13:44:48.375126][INFO][test_dist.py:228] - iter=8 dt=0.003105 dtf=0.000719 dtb=0.002385 loss=720.116943 sps=20614.492256
    [2024-11-28 13:44:48.379744][INFO][test_dist.py:228] - iter=9 dt=0.003057 dtf=0.000698 dtb=0.002359 loss=714.217957 sps=20937.174740
    [2024-11-28 13:44:48.384429][INFO][test_dist.py:228] - iter=10 dt=0.003083 dtf=0.000703 dtb=0.002380 loss=718.780884 sps=20760.967373
    [2024-11-28 13:44:48.390098][INFO][test_dist.py:228] - iter=11 dt=0.003931 dtf=0.000802 dtb=0.003129 loss=699.931152 sps=16281.262548
    [2024-11-28 13:44:48.395638][INFO][test_dist.py:228] - iter=12 dt=0.003698 dtf=0.000865 dtb=0.002832 loss=687.315857 sps=17308.773026
    [2024-11-28 13:44:48.400852][INFO][test_dist.py:228] - iter=13 dt=0.003502 dtf=0.000792 dtb=0.002709 loss=685.231995 sps=18276.152786
    [2024-11-28 13:44:48.406145][INFO][test_dist.py:228] - iter=14 dt=0.003501 dtf=0.000788 dtb=0.002714 loss=678.643860 sps=18278.977230
    [2024-11-28 13:44:48.411186][INFO][test_dist.py:228] - iter=15 dt=0.003366 dtf=0.000760 dtb=0.002606 loss=675.734375 sps=19012.740109
    [2024-11-28 13:44:48.416092][INFO][test_dist.py:228] - iter=16 dt=0.003308 dtf=0.000764 dtb=0.002545 loss=656.803894 sps=19345.774997
    [2024-11-28 13:44:48.421088][INFO][test_dist.py:228] - iter=17 dt=0.003326 dtf=0.000800 dtb=0.002526 loss=663.846558 sps=19240.551489
    [2024-11-28 13:44:48.426305][INFO][test_dist.py:228] - iter=18 dt=0.003439 dtf=0.000805 dtb=0.002634 loss=646.870605 sps=18607.582468
    [2024-11-28 13:44:48.431107][INFO][test_dist.py:228] - iter=19 dt=0.003217 dtf=0.000730 dtb=0.002488 loss=631.541504 sps=19893.098888
    [2024-11-28 13:44:48.436171][INFO][test_dist.py:228] - iter=20 dt=0.003447 dtf=0.000707 dtb=0.002741 loss=645.819580 sps=18564.526433
    [2024-11-28 13:44:48.441333][INFO][test_dist.py:228] - iter=21 dt=0.003465 dtf=0.000921 dtb=0.002543 loss=642.620361 sps=18470.919500
    [2024-11-28 13:44:48.446923][INFO][test_dist.py:228] - iter=22 dt=0.003732 dtf=0.000822 dtb=0.002910 loss=639.229980 sps=17149.966074
    [2024-11-28 13:44:48.451965][INFO][test_dist.py:228] - iter=23 dt=0.003386 dtf=0.000778 dtb=0.002608 loss=627.894897 sps=18902.966756
    [2024-11-28 13:44:48.457006][INFO][test_dist.py:228] - iter=24 dt=0.003367 dtf=0.000807 dtb=0.002560 loss=604.797424 sps=19007.381386
    [2024-11-28 13:44:48.461850][INFO][test_dist.py:228] - iter=25 dt=0.003164 dtf=0.000721 dtb=0.002443 loss=614.561523 sps=20229.778931
    [2024-11-28 13:44:48.466983][INFO][test_dist.py:228] - iter=26 dt=0.003473 dtf=0.000759 dtb=0.002714 loss=617.550781 sps=18428.700435
    [2024-11-28 13:44:48.471925][INFO][test_dist.py:228] - iter=27 dt=0.003289 dtf=0.000758 dtb=0.002531 loss=618.434082 sps=19457.441792
    [2024-11-28 13:44:48.477516][INFO][test_dist.py:228] - iter=28 dt=0.003872 dtf=0.000930 dtb=0.002942 loss=607.261475 sps=16528.336617
    [2024-11-28 13:44:48.482581][INFO][test_dist.py:228] - iter=29 dt=0.003365 dtf=0.000773 dtb=0.002591 loss=601.454590 sps=19021.673645
    [2024-11-28 13:44:48.487622][INFO][test_dist.py:228] - iter=30 dt=0.003288 dtf=0.000825 dtb=0.002463 loss=594.649170 sps=19463.454229
    [2024-11-28 13:44:48.492493][INFO][test_dist.py:228] - iter=31 dt=0.003211 dtf=0.000734 dtb=0.002477 loss=582.087036 sps=19933.062380
    [2024-11-28 13:44:48.497510][INFO][test_dist.py:228] - iter=32 dt=0.003360 dtf=0.000851 dtb=0.002509 loss=582.850586 sps=19050.324061
    [2024-11-28 13:44:48.502534][INFO][test_dist.py:228] - iter=33 dt=0.003299 dtf=0.000751 dtb=0.002547 loss=574.619019 sps=19402.524122
    [2024-11-28 13:44:48.507516][INFO][test_dist.py:228] - iter=34 dt=0.003318 dtf=0.000843 dtb=0.002475 loss=571.530273 sps=19291.396984
    [2024-11-28 13:44:48.512324][INFO][test_dist.py:228] - iter=35 dt=0.003162 dtf=0.000718 dtb=0.002444 loss=569.056335 sps=20239.766403
    [2024-11-28 13:44:48.517314][INFO][test_dist.py:228] - iter=36 dt=0.003338 dtf=0.000803 dtb=0.002535 loss=570.773315 sps=19171.898987
    [2024-11-28 13:44:48.522111][INFO][test_dist.py:228] - iter=37 dt=0.003111 dtf=0.000712 dtb=0.002399 loss=564.691101 sps=20571.738756
    [2024-11-28 13:44:48.527333][INFO][test_dist.py:228] - iter=38 dt=0.003581 dtf=0.000698 dtb=0.002883 loss=555.157349 sps=17870.995302
    [2024-11-28 13:44:48.532827][INFO][test_dist.py:228] - iter=39 dt=0.003596 dtf=0.000841 dtb=0.002755 loss=542.374756 sps=17796.934508
    [2024-11-28 13:44:48.538391][INFO][test_dist.py:228] - iter=40 dt=0.003673 dtf=0.000872 dtb=0.002801 loss=538.616821 sps=17424.614226
    [2024-11-28 13:44:48.543835][INFO][test_dist.py:228] - iter=41 dt=0.003606 dtf=0.000837 dtb=0.002769 loss=546.055054 sps=17746.977167
    [2024-11-28 13:44:48.548728][INFO][test_dist.py:228] - iter=42 dt=0.003171 dtf=0.000786 dtb=0.002385 loss=541.741455 sps=20183.008810
    [2024-11-28 13:44:48.553392][INFO][test_dist.py:228] - iter=43 dt=0.003091 dtf=0.000675 dtb=0.002415 loss=541.895630 sps=20707.731509
    [2024-11-28 13:44:48.558674][INFO][test_dist.py:228] - iter=44 dt=0.003598 dtf=0.000797 dtb=0.002801 loss=532.636841 sps=17789.399613
    [2024-11-28 13:44:48.563691][INFO][test_dist.py:228] - iter=45 dt=0.003305 dtf=0.000726 dtb=0.002579 loss=527.679077 sps=19363.041186
    [2024-11-28 13:44:48.568927][INFO][test_dist.py:228] - iter=46 dt=0.003472 dtf=0.000776 dtb=0.002696 loss=519.220581 sps=18435.547756
    [2024-11-28 13:44:48.573801][INFO][test_dist.py:228] - iter=47 dt=0.003203 dtf=0.000723 dtb=0.002480 loss=527.749268 sps=19982.521481
    [2024-11-28 13:44:48.578761][INFO][test_dist.py:228] - iter=48 dt=0.003253 dtf=0.000782 dtb=0.002471 loss=524.344238 sps=19672.846752
    [2024-11-28 13:44:48.583369][INFO][test_dist.py:228] - iter=49 dt=0.003042 dtf=0.000652 dtb=0.002390 loss=514.100464 sps=21038.853899
    [2024-11-28 13:44:48.588292][INFO][test_dist.py:228] - iter=50 dt=0.003210 dtf=0.000763 dtb=0.002447 loss=513.998962 sps=19936.401969
    [2024-11-28 13:44:48.593155][INFO][test_dist.py:228] - iter=51 dt=0.003210 dtf=0.000763 dtb=0.002447 loss=506.444519 sps=19937.409848
    [2024-11-28 13:44:48.598482][INFO][test_dist.py:228] - iter=52 dt=0.003648 dtf=0.000781 dtb=0.002868 loss=505.063721 sps=17542.989327
    [2024-11-28 13:44:48.603395][INFO][test_dist.py:228] - iter=53 dt=0.003258 dtf=0.000742 dtb=0.002516 loss=500.011047 sps=19642.561466
    [2024-11-28 13:44:48.608385][INFO][test_dist.py:228] - iter=54 dt=0.003292 dtf=0.000797 dtb=0.002495 loss=508.445740 sps=19439.362133
    [2024-11-28 13:44:48.613115][INFO][test_dist.py:228] - iter=55 dt=0.003151 dtf=0.000699 dtb=0.002452 loss=492.626648 sps=20310.433009
    [2024-11-28 13:44:48.618036][INFO][test_dist.py:228] - iter=56 dt=0.003251 dtf=0.000727 dtb=0.002524 loss=487.402435 sps=19684.735824
    [2024-11-28 13:44:48.623122][INFO][test_dist.py:228] - iter=57 dt=0.003449 dtf=0.000979 dtb=0.002469 loss=474.962097 sps=18556.851343
    [2024-11-28 13:44:48.628167][INFO][test_dist.py:228] - iter=58 dt=0.003343 dtf=0.000811 dtb=0.002532 loss=479.064941 sps=19143.536594
    [2024-11-28 13:44:48.633070][INFO][test_dist.py:228] - iter=59 dt=0.003256 dtf=0.000787 dtb=0.002468 loss=471.197083 sps=19658.706785
    [2024-11-28 13:44:48.638373][INFO][test_dist.py:228] - iter=60 dt=0.003548 dtf=0.000853 dtb=0.002696 loss=469.964081 sps=18037.965649
    [2024-11-28 13:44:48.643270][INFO][test_dist.py:228] - iter=61 dt=0.003225 dtf=0.000736 dtb=0.002489 loss=476.972076 sps=19844.166872
    [2024-11-28 13:44:48.648256][INFO][test_dist.py:228] - iter=62 dt=0.003214 dtf=0.000745 dtb=0.002469 loss=463.572174 sps=19912.478524
    [2024-11-28 13:44:48.652939][INFO][test_dist.py:228] - iter=63 dt=0.003095 dtf=0.000683 dtb=0.002411 loss=462.910156 sps=20679.849741
    [2024-11-28 13:44:48.657892][INFO][test_dist.py:228] - iter=64 dt=0.003239 dtf=0.000746 dtb=0.002492 loss=457.325439 sps=19762.175344
    [2024-11-28 13:44:48.662903][INFO][test_dist.py:228] - iter=65 dt=0.003293 dtf=0.000729 dtb=0.002563 loss=453.347168 sps=19438.021843
    [2024-11-28 13:44:48.667970][INFO][test_dist.py:228] - iter=66 dt=0.003276 dtf=0.000788 dtb=0.002488 loss=450.351135 sps=19534.356115
    [2024-11-28 13:44:48.672824][INFO][test_dist.py:228] - iter=67 dt=0.003192 dtf=0.000735 dtb=0.002457 loss=450.714233 sps=20047.789409
    [2024-11-28 13:44:48.677882][INFO][test_dist.py:228] - iter=68 dt=0.003330 dtf=0.000799 dtb=0.002530 loss=440.284546 sps=19220.840096
    [2024-11-28 13:44:48.682532][INFO][test_dist.py:228] - iter=69 dt=0.003081 dtf=0.000663 dtb=0.002418 loss=444.536011 sps=20770.230742
    [2024-11-28 13:44:48.687492][INFO][test_dist.py:228] - iter=70 dt=0.003307 dtf=0.000766 dtb=0.002541 loss=446.201233 sps=19354.965715
    [2024-11-28 13:44:48.692350][INFO][test_dist.py:228] - iter=71 dt=0.003225 dtf=0.000737 dtb=0.002488 loss=427.167328 sps=19845.047963
    [2024-11-28 13:44:48.697422][INFO][test_dist.py:228] - iter=72 dt=0.003415 dtf=0.000881 dtb=0.002534 loss=434.620087 sps=18740.536560
    [2024-11-28 13:44:48.702393][INFO][test_dist.py:228] - iter=73 dt=0.003299 dtf=0.000740 dtb=0.002559 loss=425.558777 sps=19402.652860
    [2024-11-28 13:44:48.707240][INFO][test_dist.py:228] - iter=74 dt=0.003180 dtf=0.000789 dtb=0.002391 loss=428.168945 sps=20126.919392
    [2024-11-28 13:44:48.712103][INFO][test_dist.py:228] - iter=75 dt=0.003322 dtf=0.000680 dtb=0.002642 loss=417.153503 sps=19263.558998
    [2024-11-28 13:44:48.717120][INFO][test_dist.py:228] - iter=76 dt=0.003405 dtf=0.000787 dtb=0.002618 loss=412.435059 sps=18794.204421
    [2024-11-28 13:44:48.722012][INFO][test_dist.py:228] - iter=77 dt=0.003208 dtf=0.000722 dtb=0.002487 loss=426.690186 sps=19947.873515
    [2024-11-28 13:44:48.727040][INFO][test_dist.py:228] - iter=78 dt=0.003330 dtf=0.000775 dtb=0.002554 loss=404.138733 sps=19220.615648
    [2024-11-28 13:44:48.731946][INFO][test_dist.py:228] - iter=79 dt=0.003218 dtf=0.000757 dtb=0.002461 loss=396.998413 sps=19886.330391
    [2024-11-28 13:44:48.736956][INFO][test_dist.py:228] - iter=80 dt=0.003362 dtf=0.000818 dtb=0.002544 loss=405.073059 sps=19036.951133
    [2024-11-28 13:44:48.742154][INFO][test_dist.py:228] - iter=81 dt=0.003309 dtf=0.000840 dtb=0.002468 loss=408.205780 sps=19341.447610
    [2024-11-28 13:44:48.747100][INFO][test_dist.py:228] - iter=82 dt=0.003301 dtf=0.000801 dtb=0.002500 loss=391.203918 sps=19389.697222
    [2024-11-28 13:44:48.752235][INFO][test_dist.py:228] - iter=83 dt=0.003488 dtf=0.000732 dtb=0.002756 loss=401.911407 sps=18347.650097
    [2024-11-28 13:44:48.757468][INFO][test_dist.py:228] - iter=84 dt=0.003567 dtf=0.000886 dtb=0.002681 loss=390.872192 sps=17942.862497
    [2024-11-28 13:44:48.762431][INFO][test_dist.py:228] - iter=85 dt=0.003272 dtf=0.000769 dtb=0.002502 loss=390.741089 sps=19562.065370
    [2024-11-28 13:44:48.767527][INFO][test_dist.py:228] - iter=86 dt=0.003390 dtf=0.000795 dtb=0.002596 loss=393.982513 sps=18877.408330
    [2024-11-28 13:44:48.772363][INFO][test_dist.py:228] - iter=87 dt=0.003198 dtf=0.000727 dtb=0.002471 loss=378.682495 sps=20014.348794
    [2024-11-28 13:44:48.777779][INFO][test_dist.py:228] - iter=88 dt=0.003666 dtf=0.000914 dtb=0.002752 loss=378.739502 sps=17459.234394
    [2024-11-28 13:44:48.783160][INFO][test_dist.py:228] - iter=89 dt=0.003529 dtf=0.000832 dtb=0.002697 loss=382.028931 sps=18136.158201
    [2024-11-28 13:44:48.788498][INFO][test_dist.py:228] - iter=90 dt=0.003442 dtf=0.000809 dtb=0.002633 loss=371.271118 sps=18594.284077
    [2024-11-28 13:44:48.793524][INFO][test_dist.py:228] - iter=91 dt=0.003358 dtf=0.000754 dtb=0.002603 loss=371.135925 sps=19060.348912
    [2024-11-28 13:44:48.798593][INFO][test_dist.py:228] - iter=92 dt=0.003336 dtf=0.000791 dtb=0.002545 loss=368.860168 sps=19183.369504
    [2024-11-28 13:44:48.803491][INFO][test_dist.py:228] - iter=93 dt=0.003227 dtf=0.000725 dtb=0.002502 loss=371.283356 sps=19831.370522
    [2024-11-28 13:44:48.808397][INFO][test_dist.py:228] - iter=94 dt=0.003250 dtf=0.000769 dtb=0.002481 loss=362.983154 sps=19693.397864
    [2024-11-28 13:44:48.813075][INFO][test_dist.py:228] - iter=95 dt=0.003098 dtf=0.000703 dtb=0.002396 loss=365.535828 sps=20656.015873
    [2024-11-28 13:44:48.818465][INFO][test_dist.py:228] - iter=96 dt=0.003581 dtf=0.000872 dtb=0.002709 loss=344.663239 sps=17873.310048
    [2024-11-28 13:44:48.823831][INFO][test_dist.py:228] - iter=97 dt=0.003509 dtf=0.000834 dtb=0.002675 loss=361.620361 sps=18238.825661
    [2024-11-28 13:44:48.828930][INFO][test_dist.py:228] - iter=98 dt=0.003326 dtf=0.000804 dtb=0.002522 loss=347.258301 sps=19245.190902
    [2024-11-28 13:44:48.834103][INFO][test_dist.py:228] - iter=99 dt=0.003474 dtf=0.000740 dtb=0.002733 loss=346.517212 sps=18424.412945
    Failed to download font: IBM Plex Sans, skipping!
    Failed to download font: IBM Plex Sans Condensed, skipping!
    Failed to download font: IBM Plex Serif, skipping!
    [2024-11-28 13:44:50.885911][INFO][history.py:696] - Saving train_iter plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/mplot
    [2024-11-28 13:44:51.498182][INFO][history.py:696] - Saving train_dt plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/mplot
    [2024-11-28 13:44:51.758706][INFO][history.py:696] - Saving train_dtf plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/mplot
    [2024-11-28 13:44:52.022463][INFO][history.py:696] - Saving train_dtb plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/mplot
    [2024-11-28 13:44:52.309521][INFO][history.py:696] - Saving train_loss plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/mplot
    [2024-11-28 13:44:52.562425][INFO][history.py:696] - Saving train_sps plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/mplot
                            train_iter [2024-11-28-134452]
        ┌─────────────────────────────────────────────────────────────────────┐
    99.0┤                                                                 ▗▄▄▀│
        │                                                              ▄▞▀▘   │
        │                                                          ▄▄▀▀       │
    82.7┤                                                      ▄▄▀▀           │
        │                                                  ▗▄▞▀               │
        │                                              ▄▄▀▀▘                  │
    66.3┤                                          ▗▄▀▀                       │
        │                                      ▗▄▞▀▘                          │
    50.0┤                                  ▗▄▞▀▘                              │
        │                               ▄▄▀▘                                  │
        │                          ▗▄▞▀▀                                      │
    33.7┤                       ▄▞▀▘                                          │
        │                   ▄▄▀▀                                              │
        │              ▗▄▄▀▀                                                  │
    17.3┤           ▄▄▞▘                                                      │
        │       ▄▄▀▀                                                          │
        │   ▗▄▀▀                                                              │
     1.0┤▄▞▀▘                                                                 │
        └─┬─┬───┬──┬──┬────┬──┬──┬────┬───┬──┬──┬──┬───┬──┬───┬──┬──┬──┬────┬─┘
          2 5  11 16 20   27 31 36   43  48 53 57 61  68 71  78 81 86 91   98
    train_iter                        train/iter
    [2024-11-28 13:44:52.868990][INFO][plot.py:220] - Appending plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_iter.txt
    text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_iter.txt
                              train_dt [2024-11-28-134452]
          ┌───────────────────────────────────────────────────────────────────┐
    0.0197┤▌                                                                  │
          │▌                                                                  │
          │▌                                                                  │
    0.0169┤▌                                                                  │
          │▌                                                                  │
          │▌                                                                  │
    0.0142┤▌                                                                  │
          │▌                                                                  │
    0.0114┤▌                                                                  │
          │▌                                                                  │
          │▌                                                                  │
    0.0086┤▌                                                                  │
          │▌                                                                  │
          │▌                                                                  │
    0.0058┤▌                                                                  │
          │▌                                                                  │
          │▚      ▖          ▗                                                │
    0.0030┤ ▚▄▄▄▄▞▝▀▀▀▀▄▀▀▀▄▀▘▀▄▀▀▄▚▀▀▚▄▀▀▄▄▄▞▄▚▄▀▄▚▄▄▞▄▞▄▚▀▚▀▚▞▞▀▀▀▞▄▀▀▀▚▄▞▀▀│
          └─┬─┬───┬──┬──┬────┬──┬──┬───┬───┬──┬────┬────┬───┬────┬──┬───┬───┬─┘
            2 5  11 16 20   27 32 36  43  48 53   61   68  74   81 86  91  98
    train_dt                           train/iter
    [2024-11-28 13:44:52.888254][INFO][plot.py:220] - Appending plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_dt.txt
    text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_dt.txt
                               train_dtf [2024-11-28-134452]
           ┌──────────────────────────────────────────────────────────────────┐
    0.00296┤▌                                                                 │
           │▌                                                                 │
           │▌                                                                 │
    0.00257┤▌                                                                 │
           │▌                                                                 │
           │▌                                                                 │
    0.00219┤▌                                                                 │
           │▌                                                                 │
    0.00181┤▌                                                                 │
           │▌                                                                 │
           │▌                                                                 │
    0.00142┤▌                                                                 │
           │▌                                                                 │
           │▌                                                                 │
    0.00104┤▌                                                                 │
           │▌            ▗    ▖                  ▟                 ▗  ▖       │
           │▝▖▗   ▗▞▄▄▄▄▗▘▚▄▗▟▚▟▞▄▌▗ ▞▀▄▗ ▖▗ ▄▖▗ ▛▄▟   ▗ ▖▖▟ ▖▖▗▄▀▖▛▄▟▝▚▄▖▖▞▚▖│
    0.00065┤ ▝▘▀▀▀▘     ▀   ▘   ▘ ▝▘▀  ▝▌▀▝▀▞ ▝▘▀▘  ▀▚▀▘▀▜▝▘▀▜▝▘  ▝▘ ▝   ▝▝▘ ▝│
           └─┬─┬───┬──┬──┬───┬──┬──┬────┬──┬──┬──┬──┬───┬───┬────┬──┬───┬───┬─┘
             2 5  11 16 20  27 31 36   43 48 53 57 61  68  74   81 86  91  98
    train_dtf                           train/iter
    [2024-11-28 13:44:52.904574][INFO][plot.py:220] - Appending plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_dtf.txt
    text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_dtf.txt
                              train_dtb [2024-11-28-134452]
          ┌───────────────────────────────────────────────────────────────────┐
    0.0168┤▌                                                                  │
          │▌                                                                  │
          │▌                                                                  │
    0.0144┤▌                                                                  │
          │▌                                                                  │
          │▌                                                                  │
    0.0120┤▌                                                                  │
          │▌                                                                  │
    0.0096┤▌                                                                  │
          │▌                                                                  │
          │▌                                                                  │
    0.0072┤▌                                                                  │
          │▌                                                                  │
          │▌                                                                  │
    0.0048┤▌                                                                  │
          │▌                                                                  │
          │▚      ▖                                                           │
    0.0024┤ ▚▄▄▄▄▞▝▀▀▄▞▄▚▞▚▄▚▞▚▄▄▄▄▄▀▀▚▄▀▀▄▄▄▞▄▄▄▄▄▚▄▄▄▄▄▄▄▄▄▀▚▄▄▄▞▀▞▄▀▀▚▄▄▞▚▞│
          └─┬─┬───┬──┬──┬────┬──┬──┬───┬───┬──┬────┬────┬───┬────┬──┬───┬───┬─┘
            2 5  11 16 20   27 32 36  43  48 53   61   68  74   81 86  91  98
    train_dtb                          train/iter
    [2024-11-28 13:44:52.920733][INFO][plot.py:220] - Appending plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_dtb.txt
    text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_dtb.txt
                             train_loss [2024-11-28-134452]
          ┌───────────────────────────────────────────────────────────────────┐
    1821.1┤▌                                                                  │
          │▌                                                                  │
          │▌                                                                  │
    1575.1┤▌                                                                  │
          │▌                                                                  │
          │▚                                                                  │
    1329.0┤▐                                                                  │
          │▝▖                                                                 │
    1082.9┤ ▌                                                                 │
          │ ▐                                                                 │
          │ ▝▖                                                                │
     836.8┤  ▚                                                                │
          │   ▀▖                                                              │
          │    ▝▀▚▄▄▄ ▖                                                       │
     590.7┤          ▀▝▀▀▀▚▄▄▄▄▄▖                                             │
          │                     ▝▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄                              │
          │                                     ▀▀▀▀▀▀▀▀▚▄▄▄▄▄▄  ▗            │
     344.7┤                                                    ▀▀▘▀▀▀▀▀▀▀▀▄▄▄▄│
          └─┬─┬───┬──┬──┬────┬──┬──┬───┬───┬──┬────┬────┬───┬────┬──┬───┬───┬─┘
            2 5  11 16 20   27 32 36  43  48 53   61   68  74   81 86  91  98
    train_loss                         train/iter
    [2024-11-28 13:44:52.939625][INFO][plot.py:220] - Appending plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_loss.txt
    text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_loss.txt
                               train_sps [2024-11-28-134452]
           ┌──────────────────────────────────────────────────────────────────┐
    21038.9┤  ▖▗▄▞▖                 ▖   ▖  ▗▚   ▖    ▗   ▗                 ▖  │
           │ ▐▝▘  ▌   ▖▗▌  ▗▌▗ ▗▌▗▞▟▌  ▞▌▗▞▟ ▜ ▟▝▖ ▄▀▀▄▄▚▛▄▌▄▚▗▙▚▗▖ ▖▟   ▞▞▌  │
           │ ▌    ▌ ▄▞▝▟▚▄▞▀▚▜▞▘▝▘ ▝▌ ▗▘▙▀▌  ▐▐  ▚▀█     ▘ ▜  ▘▝ ▘▝▟▝▜ ▗▀▘ ▌▗▚│
    18073.5┤ ▌    ▌▐     ▐▌  ▐▌     ▝▀▞ ▝    ▝▌    ▝               ▝ ▝▄▘   ▝▘ │
           │▐     ▚▘      ▘  ▝▌                                               │
           │▐                                                                 │
    15108.1┤▌                                                                 │
           │▌                                                                 │
    12142.7┤▌                                                                 │
           │▌                                                                 │
           │▌                                                                 │
     9177.3┤▌                                                                 │
           │▌                                                                 │
           │▌                                                                 │
     6211.9┤▌                                                                 │
           │▌                                                                 │
           │▌                                                                 │
     3246.6┤▌                                                                 │
           └─┬─┬───┬──┬──┬───┬──┬──┬────┬──┬──┬──┬──┬───┬───┬────┬──┬───┬───┬─┘
             2 5  11 16 20  27 31 36   43 48 53 57 61  68  74   81 86  91  98
    train_sps                           train/iter
    [2024-11-28 13:44:52.955997][INFO][plot.py:220] - Appending plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_sps.txt
    text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/test-dist-plots/tplot/train_sps.txt
    [2024-11-28 13:44:53.002178][INFO][test_dist.py:246] - dataset=<xarray.Dataset> Size: 5kB
    Dimensions:     (draw: 99)
    Coordinates:
      * draw        (draw) int64 792B 0 1 2 3 4 5 6 7 8 ... 91 92 93 94 95 96 97 98
    Data variables:
        train_iter  (draw) int64 792B 1 2 3 4 5 6 7 8 9 ... 92 93 94 95 96 97 98 99
        train_dt    (draw) float64 792B 0.01971 0.004083 ... 0.003326 0.003474
        train_dtf   (draw) float64 792B 0.002959 0.0008221 ... 0.0008038 0.0007404
        train_dtb   (draw) float64 792B 0.01675 0.003261 ... 0.002522 0.002733
        train_loss  (draw) float32 396B 1.821e+03 1.348e+03 ... 347.3 346.5
        train_sps   (draw) float64 792B 3.247e+03 1.567e+04 ... 1.925e+04 1.842e+04

      _     ._   __/__   _ _  _  _ _/_   Recorded: 13:44:35  Samples:  2127
     /_//_/// /_\ / //_// / //_'/ //     Duration: 17.440    CPU time: 29.088
    /   _/                      v5.0.0

    Profile at /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/src/ezpz/profile.py:101

    17.440 <module>  ezpz/test_dist.py:1
    └─ 17.440 main  ezpz/test_dist.py:177
       ├─ 10.558 build_model_and_optimizer  ezpz/test_dist.py:136
       │  └─ 10.547 DistributedDataParallel.__init__  torch/nn/parallel/distributed.py:622
       │     └─ 10.502 _verify_param_shape_across_processes  torch/distributed/utils.py:266
       │        └─ 10.502 PyCapsule._verify_params_across_processes  <built-in>
       ├─ 3.256 History.plot_all  ezpz/history.py:636
       │  ├─ 1.809 savefig  matplotlib/pyplot.py:974
       │  │     [38 frames hidden]  matplotlib, PIL, <built-in>, numpy
       │  ├─ 0.895 <module>  seaborn/__init__.py:1
       │  │     [8 frames hidden]  seaborn, scipy
       │  ├─ 0.227 History.get_dataset  ezpz/history.py:752
       │  │  └─ 0.183 History.to_DataArray  ezpz/history.py:712
       │  │     └─ 0.183 DataArray.__init__  xarray/core/dataarray.py:437
       │  │           [5 frames hidden]  xarray
       │  └─ 0.179 make_ridgeplots  ezpz/plot.py:900
       ├─ 2.016 _forward_step  ezpz/test_dist.py:193
       │  └─ 1.963 DistributedDataParallel._wrapped_call_impl  torch/nn/modules/module.py:1528
       │        [4 frames hidden]  torch
       │           1.945 Network._call_impl  torch/nn/modules/module.py:1534
       │           └─ 1.943 Network.forward  ezpz/test_dist.py:128
       │              └─ 1.943 Sequential._wrapped_call_impl  torch/nn/modules/module.py:1528
       │                    [6 frames hidden]  torch, <built-in>
       ├─ 0.716 <module>  ambivalent/__init__.py:1
       │     [20 frames hidden]  ambivalent, requests, urllib3, http, ...
       └─ 0.495 _backward_step  ezpz/test_dist.py:198
          └─ 0.457 Tensor.backward  torch/_tensor.py:466
                [3 frames hidden]  torch, <built-in>


    [2024-11-28 13:44:54.356012][INFO][profile.py:115] - Saving pyinstrument profile output to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/ezpz_pyinstrument_profiles
    [2024-11-28 13:44:54.356642][INFO][profile.py:123] - PyInstrument profile saved (as html) to:  /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/ezpz_pyinstrument_profiles/pyinstrument-profile-2024-11-28-134454.html
    [2024-11-28 13:44:54.357100][INFO][profile.py:131] - PyInstrument profile saved (as text) to:  /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/ezpz_pyinstrument_profiles/pyinstrument-profile-2024-11-28-134454.txt
    [2024-11-28 13:44:54.716119][INFO][profile.py:143] - Finished with pyinstrument profiler. Took: 17.44008s
    [2024-11-28 13:44:54.717891][INFO][test_dist.py:269] - [0] runtime=73.419312s
    wandb: 🚀 View run driven-wind-683 at: https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/4mwyy84l
    wandb: Find logs at: ../../../../../../lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/ezpz/wandb/run-20241128_134434-4mwyy84l/logs
    Application ecd9868b resources: utime=2078s stime=374s maxrss=2509188KB inblock=786752 oublock=12104 minflt=13378945 majflt=181980 nvcsw=1704995 nivcsw=216931
    took: 0h:01m:32s
    ```

    </details>


