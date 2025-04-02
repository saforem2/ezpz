# 🐍 Python Library

### 👀 Overview

Launch and train across all your accelerators, using your favorite framework +
backend combo.

`ezpz` simplifies the process of:

- <details><summary>Setting up + launching distributed training:</summary>

    - <details closed><summary><code>import ezpz as ez</code></summary>

        - `RANK = `
          [`ez.setup_torch(backend=backend)`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L551)
          [for `backend` $\in$ \{`DDP`, `deepspeed`, `horovod`}]{.dim-text}

        - `RANK =`
          [`ez.get_rank()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#396)

        - `LOCAL_RANK =`
          [`ez.get_local_rank()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#448)

        - `WORLD_SIZE =`
          [`ez.get_world_size()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L417)

        [(see [`ezpz/dist.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py) for more details).]{.dim-text}

    </details>

</details>

- <details closed><summary>Using your favorite framework:</summary>

    - `framework=pytorch` + `backend={DDP, deepspeed, horovod}`

    - `framework=tensorflow` + `backend=horovod`

    - [`ez.get_torch_device()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L332): {`cuda`, `xpu`, `mps`, `cpu`}

    - [`ez.get_torch_backend()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L348): {`nccl`, `ccl`, `gloo`}

  _2ez_ 😎. (see [frameworks](#frameworks) for additional details)

</details>

- <details closed><summary>Writing device agnostic code:</summary>

    - <details><summary><a href="https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L332"><code>ezpz.get_torch_device()</code></a></summary>

        ```python
        >>> import ezpz as ez
        >>> DEVICE = ez.get_torch_device()
        >>> model = torch.nn.Linear(10, 10)
        >>> model.to(DEVICE)
        >>> x = torch.randn((10, 10), device=DEVICE)
        >>> y = model(x)
        >>> y.device
        device(type='mps', index=0)
        ```

    </details>

</details>

- <details closed><summary>Using <code>wandb</code>:</summary>

    - `ez.setup_wandb(project_name='ezpz')`

</details>

- **Full support** for any {`device` + `framework` + `backend`}:
    - device: {`GPU`, `XPU`, `MPS`, `CPU`}
    - framework: {`torch`, `deepspeed`, `horovod`, `tensorflow`}
    - backend: {`DDP`, `deepspeed`, `horovod`}
 
### Install

To install[^5]:

``` bash
python3 -m pip install -e "git+https://github.com/saforem2/ezpz#egg=ezpz" --require-virtualenv
```

[^5]: Note the `--require-virtualenv` isn’t *strictly* required, but I
    highly recommend to always try and work within a virtual
    environment, when possible.


- 📂 `ezpz` / `src` / [`ezpz/`](https://github.com/saforem2/ezpz)
  - 📂
    [`bin/`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/):
    - [`utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh):
      Shell utilities for `ezpz`
  - 📂
    [`conf/`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/conf/):
    - ⚙️
      [`config.yaml`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/conf/config.yaml):
      Default `TrainConfig` object
    - ⚙️
      [`ds_config.json`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/conf/ds_config.json):
      DeepSpeed configuration
  - 📂
    [`log/`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/log/):
    Logging configuration.
  - 🐍
    [`__about__.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/__about__.py):
    Version information
  - 🐍
    [`__init__.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/__init__.py):
    Main module
  - 🐍
    [`__main__.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/__main__.py):
    Entry point
  - 🐍
    [`configs.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/configs.py):
    Configuration module
  - 🐍[`cria.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/create.py):
    Baby Llama
  - 🐍[**`dist.py`**](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py):
    Distributed training module
  - 🐍[`history.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/history.py):
    History module
  - 🐍[`jobs.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/jobs.py):
    Jobs module
  - 🐍[`model.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/model.py):
    Model module
  - 🐍[`plot.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/plot.py):
    Plot modul
  - 🐍[`profile.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/profile.py):
    Profile module
  - 🐍[`runtime.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/runtime.py):
    Runtime module
  - 🐍[`test.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test.py):
    Test module
  - 🐍[**`test_dist.py`**](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py):
    Distributed training test module
  - 🐍[`train.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/train.py):
    train module
  - 🐍[`trainer.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/trainer.py):
    trainer module
  - 🐍[`utils.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/utils.py):
    utility module

``` bash
📂 /ezpz/src/ezpz/
┣━━ 📂 bin/
┃   ┣━━ 📄 affinity.sh
┃   ┣━━ 📄 getjobenv
┃   ┣━━ 📄 savejobenv
┃   ┣━━ 📄 saveslurmenv
┃   ┣━━ 📄 setup.sh
┃   ┣━━ 📄 train.sh
┃   ┗━━ 📄 utils.sh
┣━━ 📂 conf/
┃   ┣━━ 📂 hydra/
┃   ┃   ┗━━ 📂 job_logging/
┃   ┃       ┣━━ ⚙️ colorlog1.yaml
┃   ┃       ┣━━ ⚙️ custom.yaml
┃   ┃       ┗━━ ⚙️ enrich.yaml
┃   ┣━━ 📂 logdir/
┃   ┃   ┗━━ ⚙️ default.yaml
┃   ┣━━ ⚙️ config.yaml
┃   ┣━━ 📄 ds_config.json
┃   ┗━━ ⚙️ ds_config.yaml
┣━━ 📂 log/
┃   ┣━━ 📂 conf/
┃   ┃   ┗━━ 📂 hydra/
┃   ┃       ┗━━ 📂 job_logging/
┃   ┃           ┗━━ ⚙️ enrich.yaml
┃   ┣━━ 🐍 __init__.py
┃   ┣━━ 🐍 __main__.py
┃   ┣━━ 🐍 config.py
┃   ┣━━ 🐍 console.py
┃   ┣━━ 🐍 handler.py
┃   ┣━━ 🐍 style.py
┃   ┣━━ 🐍 test.py
┃   ┗━━ 🐍 test_log.py
┣━━ 🐍 __about__.py
┣━━ 🐍 __init__.py
┣━━ 🐍 __main__.py
┣━━ 🐍 configs.py
┣━━ 🐍 cria.py
┣━━ 🐍 dist.py
┣━━ 🐍 history.py
┣━━ 🐍 jobs.py
┣━━ 🐍 loadjobenv.py
┣━━ 🐍 model.py
┣━━ 🐍 plot.py
┣━━ 🐍 profile.py
┣━━ 🐍 runtime.py
┣━━ 🐍 savejobenv.py
┣━━ 🐍 test.py
┣━━ 🐍 test_dist.py
┣━━ 🐍 train.py
┣━━ 🐍 trainer.py
┗━━ 🐍 utils.py
```
