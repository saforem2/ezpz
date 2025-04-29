# ![🍋](./assets/lemon.svg) `ezpz`

> Train across **all** your {NVIDIA, AMD, Intel, MPS, ...} accelerators, `ezpz` 🍋.

See [🍋 `ezpz` docs](https://saforem2.github.io/ezpz) for additional information.

## 🐣Getting Started

### Example

1. 🏖️ Setup environment[^magic] (see [Shell Environment](docs/shell-environment.md)):

    ```bash
    source <(curl https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh) && ezpz_setup_env
    ```

   [^magic]:
       This will 🪄 _automagically_ source
       [`ezpz/bin/utils.sh`](src/ezpz/bin/utils.sh)
       and (`&&`) call `ezpz_setup_env` to setup your
       python environment.

1. 🐍 Install `ezpz` (see [Python Library](docs/python-library.md)):

    ```bash
    python3 -m pip install "git+https://github.com/saforem2/ezpz"
    ```

1. 🚀 Launch _any_ `*.py`[^module] **_from_** python (see [Launch](docs/launch.md)):

    ```bash
    python3 -m ezpz.launch -m ezpz.test_dist
    ```

   [^module]:
       Technically, we're _launching_ (`-m ezpz.launch`) the
       [`ezpz/test_dist.py`](src/ezpz/test_dist.py) as a module (`-m`),
       in this example.

😎 2 ez.
