# 🍋 `ezpz`

> Write _once_, run _anywhere_

Train across **all** your {NVIDIA, AMD, Intel, MPS, ...} accelerators, `ezpz` 🍋.

See [🍋 `ezpz` docs](https://saforem2.github.io/ezpz) for additional information.

## 🐣 Getting Started

1. 🏖️ Setup environment[^magic] (see [Shell Environment](./shell-environment.md)):

    ```bash
    source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env
    ```

   [^magic]:
       This will 🪄 _automagically_ source
       [`ezpz/bin/utils.sh`](src/ezpz/bin/utils.sh)
       and (`&&`) call `ezpz_setup_env` to setup your
       python environment.

<!-- 1. 🐍 Install `ezpz` (see [Python Library](./python-library.md)): -->
1. 🐍 Install `ezpz` (see [💾 Code Reference / ezpz](./Code-Reference/ezpz-reference.md)

    ```bash
    python3 -m pip install "git+https://github.com/saforem2/ezpz"
    ```

1. 🚀 Launch _any_ `*.py`[^module] **_from_** python (see [Launch](./launch.md)):

    ```bash
    ezpz.launch -m ezpz.test_dist
    ```

   [^module]:
       Technically, we're _launching_ (`-m ezpz.launch`) the
       [`ezpz/test_dist.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py)
       as a module (`-m`), in this example.

😎 2 ez.
