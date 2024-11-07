# 🍋 `ezpz`

## Getting Started

There are two main, distinct components of `ezpz`:

1. [Shell interface](#shell-interface)
2. [Python Library](#python-library)

### Shell Interface

- [`bin/utils.sh`](/src/ezpz/bin/utils.sh):
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
