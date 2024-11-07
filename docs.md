# 🍋 `ezpz`

## 🚀 Getting Started

```bash
git clone https://github.com/saforem2/ezpz deps/ezpz
```

### Using Bash Helper Functions

We provide a variety of useful functions in [`utils.sh`](/ezpz/src/ezpz/bin/utils.sh)

```bash
git clone https://github.com/saforem2/ezpz deps/ezpz
source deps/ezpz/src/ezpz/bin/utils.sh
```

- 🐍 Setup Python:

    ```bash
    ezpz_setup_python
    ```

- 🛰️ Setup communications:

    ```bash
    ezpz_setup_job
    ```

> [!WARNING]<br> **Where am I?**
> 
> _Some_ of the `ezpz_*` functions
> (e.g. `ezpz_setup_python`),
> will try to create / look for certain directories.
>
> In an effort to be explicit,
> these directories will be defined
> **relative to** a `WORKING_DIR`
> (e.g. `"${WORKING_DIR}/venvs/"`)
>
> This `WORKING_DIR` will be assigned to the first non-zero match found below:
> 
> 1. `PBS_O_WORKDIR`: If found in environment, paths will be relative to this
> 2. `SLURM_SUBMIT_DIR`: Next in line. If not @ ALCF, maybe using `slurm`...
> 3. `$(pwd)`: Otherwise, no worries. Use your _actual_ working directory.

## 🔍 In Detail

There are two main, distinct components of `ezpz`:

1. 🐚 [Shell interface](#shell-interface)
2. 🐍 [Python Library](#python-library)

### Shell Interface

> [!IMPORTANT]
> All `ezpz_*` helper functions are defined in:
> 
> [`ezpz/src/ezpz/bin/utils.sh`](/ezpz/src/ezpz/bin/utils.sh)

- Provides various (`bash` / shell) helper functions to make life easy
- Designed to be `source`-d, e.g.

   ```bash
   source ezpz/src/ezpz/bin/utils.sh
   ```
   
- All functions prefixed with `ezpz_`

To use:

```bash
git clone https://github.com/saforem2/ezpz deps/ezpz
# on ALCF:
export PBS_O_WORKDIR=$(pwd)
source deps/ezpz/src/ezpz/bin/utils.sh
ezpz_setup_python
# from a compute node:
ezpz_setup_job
```

### Python Library

```bash
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
