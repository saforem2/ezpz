# 🍋 `ezpz`

1. 🏖️ Setup shell environment:

   ```bash
   source <(curl https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh) && ezpz_setup_env
   ```

   this will 🪄 _automagically_ source [`ezpz/bin/utils.sh`](./src/ezpz/bin/utils.sh)
   and (`&&`) call `ezpz_setup_env`.

   See [shell-environment](/docs/shell-environment.md) for details.

2. 📦 Install `ezpz`:

   ```bash
   python3 -m pip install "git+https://github.com/saforem2/ezpz"
   ```

3. 🚀 Launch _any_ `*.py` **_from_** python (see [launch](docs/launch.md)):

    ```bash
    python3 -m ezpz.launch -m ezpz.test_dist
    ```

    [`ezpz/test_dist.py`](src/ezpz/test_dist.py), in this example.

    😎 2 ez.
