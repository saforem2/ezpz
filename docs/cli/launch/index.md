# 🚀 `ezpz launch`

Single entry point for distributed jobs.

`ezpz` detects PBS/Slurm automatically and falls back to `mpirun`, forwarding
useful environment variables so your script behaves the same on laptops and
clusters.

Add your own args to any command (`--config`, `--batch-size`, etc.) and `ezpz`
will propagate them through the detected launcher.

Use the provided:

```bash
ezpz launch <launch flags> -- <cmd> <cmd flags>
```

to automatically launch `<cmd>` across all available[^schedulers]
accelerators.


- ??? abstract "`ezpz launch --help`"

        ```bash
        ezpz launch --help
        usage: ezpz launch [-h] [--print-source] [--filter FILTER [FILTER ...]] [-n NPROC] [-ppn NPROC_PER_NODE] [-nh NHOSTS] [--hostfile HOSTFILE] ...

        Launch a command on the current PBS/SLURM job.

        Additional `<launcher flags>` can be passed through directly
        to the launcher by including '--' as a separator before
        the command.

        Examples:

            ezpz launch <launcher flags> -- <command> <args>

            ezpz launch -n 8 -ppn 4 --verbose --tag-output -- python3 -m ezpz.examples.fsdp_tp

            ezpz launch --nproc 8 -x EZPZ_LOG_LEVEL=DEBUG -- python3 my_script.py --my-arg val

        positional arguments:
        command               Command (and arguments) to execute. Use '--' to separate options when needed.

        options:
        -h, --help            show this help message and exit
        --print-source        Print the location of the launch CLI source and exit.
        --filter FILTER [FILTER ...]
                                Filter output lines by these strings.
        -n NPROC, -np NPROC, --n NPROC, --np NPROC, --nproc NPROC, --world_size NPROC, --nprocs NPROC
                                Number of processes.
        -ppn NPROC_PER_NODE, --ppn NPROC_PER_NODE, --nproc_per_node NPROC_PER_NODE
                                Processes per node.
        -nh NHOSTS, --nh NHOSTS, --nhost NHOSTS, --nnode NHOSTS, --nnodes NHOSTS, --nhosts NHOSTS, --nhosts NHOSTS
                                Number of nodes to use.
        --hostfile HOSTFILE   Hostfile to use for launching.
        ```

- **Scheduler smarts:** detects PBS/Slurm automatically;  
  Otherwise falls back to `mpirun` with sensible env forwarding.
  For launcher-only flags/env (e.g., `-x FOO=bar`), place them before `--`;
  everything after `--` is the command to run:

    ```bash
    ezpz launch <launch flags> -- <command to run> <command args>
    ```

    ??? abstract "Examples"

        e.g.:

        ```bash
        ezpz launch -- python3 -m ezpz.examples.fsdp
        ```

        or, specify `-n 8` processes, forward a specific `PYTHONPATH`, and set
        `EZPZ_LOG_LEVEL=DEBUG`:

        ```bash
        ezpz launch -n 8 \
            -x PYTHONPATH=/tmp/.venv/bin:${PYTHONPATH} \
            -x EZPZ_LOG_LEVEL=DEBUG \
            -- \
            python3 -m ezpz.examples.fsdp
        ```

- Automatic distributed initialization using
  [`ezpz.setup_torch()`](https://ezpz.cool/python/Code-Reference/dist/#ezpz.dist.setup_torch)
  with automatic {device, backend} selection

    ```python
    import ezpz
    _ = ezpz.setup_torch()

    device = ezpz.get_torch_device()
    # cuda, xpu, mps, cpu, ...
    ```

- Automatic single-process logging with rank-aware filtering for distributed
  runs:

    ```python
    logger = ezpz.get_logger(__name__)
    ```

- Metric tracking, aggregation, and recording via
  [`ezpz.History()`](https://ezpz.cool/python/Code-Reference/#ezpz.History):
    - Automatic distributed statistics (min, max, mean, stddev) across ranks[^distributed-history]
    - Weights & Biases integration
    - Persistent storage of metrics in `.h5` format
    - Plotting support:
    ??? example "Graphical plots (`svg`, `png`) via `matplotlib`"

          ![Accuracy](../../assets/mplot/svgs/accuracy.svg)
          ![Loss](../../assets/mplot/svgs/loss.svg)
          ![Forward time](../../assets/mplot/svgs/dtf.svg)
          ![Backward time](../../assets/mplot/svgs/dtb.svg)

    ??? example "Terminal-based ASCII plots via <a href="https://github.com/piccolomo/plotext"><code>plotext</code></a>"
        <div class="ansi-block">
        <pre class="terminal">
                            dt                                    dt/min
             ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
        0.384┤▌                                │0.384┤<span style='color:var(--cyan)'>-</span>                                │
        0.320┤▐                                │0.129┤ <span style='color:var(--cyan)'>--------------------------------</span>│
        0.256┤ ▚                               │     └┬───────┬───────┬───────┬───────┬┘
        0.129┤ ▝▖                              │     1.0     3.2     5.5     7.8   10.0 
        0.066┤  ▐                              │dt/min              iter
        0.002┤   ▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│                    dt/std
             └┬───────┬───────┬───────┬───────┬┘       ┌───────────────────────────────┐
             1.0     3.2     5.5     7.8   10.0 0.00068┤             <span style='color:var(--magenta)'>\*</span>      <span style='color:var(--magenta)'>\*</span>      <span style='color:var(--magenta)'>\*</span>   │
        dt                  iter                0.00046┤       <span style='color:var(--magenta)'>\*\*\*\*\*\*</span> <span style='color:var(--magenta)'>\*\*</span>   <span style='color:var(--magenta)'>\*</span> <span style='color:var(--magenta)'>\*\*\*\*\*\*</span> <span style='color:var(--magenta)'>\*\*\*</span>│
                        dt/mean                 0.00011┤<span style='color:var(--magenta)'>\*\*\*\*\*\*\*</span>         <span style='color:var(--magenta)'>\*\*\*</span>            │
             ┌─────────────────────────────────┐       └┬───────┬──────┬───────┬──────┬┘
        0.384┤<span style='color:var(--green)'>·</span>                                │       1.0     3.2    5.5     7.8  10.0 
        0.320┤<span style='color:var(--green)'>·</span>                                │dt/std               iter
        0.256┤ <span style='color:var(--green)'>·</span>                               │                   dt/max
        0.193┤  <span style='color:var(--green)'>·</span>                              │     ┌─────────────────────────────────┐
        0.129┤  <span style='color:var(--green)'>·</span>                              │0.384┤<span style='color:var(--red)'>+</span>                                │
        0.066┤   <span style='color:var(--green)'>·</span>                             │0.257┤ <span style='color:var(--red)'>++</span>                              │
        0.002┤    <span style='color:var(--green)'>·····························</span>│0.066┤   <span style='color:var(--red)'>++++++++++++++++++++++++++++++</span>│
             └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
            1.0     3.2     5.5     7.8   10.0      1.0     3.2     5.5     7.8   10.0 
        dt/mean             iter                dt/max              iter                
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/dt.txt</span>
             ┌─────────────────────────────────────────────────────────────────────────┐
        0.384┤ <span style='color:var(--red)'>++</span> dt/max                                                               │
             │ <span style='color:var(--cyan)'>--</span> dt/min                                                               │
             │ <span style='color:var(--green)'>··</span> dt/mean                                                              │
        0.320┤ ▞▞ dt                                                                   │
             │ ▐                                                                       │
             │  ▌                                                                      │
        0.256┤  ▚                                                                      │
             │  ▝▖                                                                     │
             │   ▌                                                                     │
        0.193┤   ▐                                                                     │
             │    ▌                                                                    │
             │    ▐                                                                    │
             │    ▝▖                                                                   │
        0.129┤     ▚                                                                   │
             │     ▐                                                                   │
             │      ▌                                                                  │
        0.065┤      ▐                                                                  │
             │      ▝▖                                                                 │
             │       ▚                                                                 │
        0.002┤       ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
             └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
             1.0               3.2               5.5               7.8             10.0 
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/dt_summary.txt</span>
                       dt/mean hist                             dt/max hist             
           ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
        9.0┤████                               │9.0┤████                               │
        7.5┤████                               │7.5┤████                               │
        6.0┤████                               │6.0┤████                               │
        4.5┤████                               │4.5┤████                               │
        3.0┤████                               │3.0┤████                               │
        1.5┤████                           ████│1.5┤████                           ████│
        0.0┤███                            ████│0.0┤███                            ████│
           └┬────────┬───────┬────────┬───────┬┘   └┬────────┬───────┬────────┬───────┬┘
           -0.01    0.09    0.19     0.30   0.40   -0.01    0.09    0.19     0.30   0.40 
                        dt/min hist                              dt/std hist
           ┌───────────────────────────────────┐    ┌──────────────────────────────────┐
        9.0┤████                               │2.00┤       ███                    ████│
        7.5┤████                               │1.67┤       ███                    ████│
        6.0┤████                               │1.33┤       ███                    ████│
        4.5┤████                               │1.00┤█████████████████   ████   ███████│
           │████                               │    │█████████████████   ████   ███████│
        3.0┤████                               │0.67┤█████████████████   ████   ███████│
        1.5┤████                           ████│0.33┤█████████████████   ████   ███████│
        0.0┤███                            ████│0.00┤█████████████████   ████   ███████│
           └┬────────┬───────┬────────┬───────┬┘    └┬───────┬────────┬───────┬────────┘
           -0.02    0.09    0.19     0.30   0.40   -0.00003 0.00016  0.00034 0.00053     
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/dt_hist.txt</span>
                            loss                                  loss/min              
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        43.4┤              ▗▀▀▀▀▄▄▄▄           │39.3┤    <span style='color:var(--cyan)'>-</span>          <span style='color:var(--cyan)'>------------</span>       │
        38.6┤   ▟         ▗▘        ▚▖         │22.7┤<span style='color:var(--cyan)'>----</span> <span style='color:var(--cyan)'>----------</span>            <span style='color:var(--cyan)'>-------</span>│
        33.7┤  ▞ ▚       ▗▘          ▝▚▖       │    └┬───────┬────────┬───────┬───────┬┘
        24.1┤ ▐   ▚     ▗▘             ▝▀▚▄▖   │    1.0     3.2      5.5     7.8   10.0 
        19.3┤▗▘    ▚   ▄▘                  ▝▀▀▀│loss/min            iter
        14.4┤▌      ▚▄▀                        │                  loss/std
            └┬───────┬────────┬───────┬───────┬┘    ┌──────────────────────────────────┐
            1.0     3.2      5.5     7.8   10.0 17.4┤           <span style='color:var(--magenta)'>\*</span>                      │
        loss                iter                11.8┤       <span style='color:var(--magenta)'>\*\*\*\*</span> <span style='color:var(--magenta)'>\*\*</span>               <span style='color:var(--magenta)'>\*</span>    │
                          loss/mean              3.3┤<span style='color:var(--magenta)'>\*\*\*\*\*\*\*</span>       <span style='color:var(--magenta)'>\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*</span> <span style='color:var(--magenta)'>\*\*\*\*</span>│
            ┌──────────────────────────────────┐    └┬───────┬────────┬───────┬───────┬┘
        41.4┤               <span style='color:var(--green)'>····</span>               │    1.0     3.2      5.5     7.8   10.0 
        37.6┤    <span style='color:var(--green)'>·</span>      <span style='color:var(--green)'>····</span>    <span style='color:var(--green)'>····</span>           │loss/std            iter                
        33.9┤   <span style='color:var(--green)'>·</span> <span style='color:var(--green)'>·</span>    <span style='color:var(--green)'>·</span>            <span style='color:var(--green)'>·······</span>    │                  loss/max              
        30.2┤  <span style='color:var(--green)'>·</span>   <span style='color:var(--green)'>·</span>  <span style='color:var(--green)'>·</span>                    <span style='color:var(--green)'>··</span>  │    ┌──────────────────────────────────┐
        26.4┤ <span style='color:var(--green)'>·</span>     <span style='color:var(--green)'>··</span>                       <span style='color:var(--green)'>··</span>│56.3┤           <span style='color:var(--red)'>+</span>                      │
        22.7┤<span style='color:var(--green)'>·</span>                                 │45.3┤    <span style='color:var(--red)'>+++++++</span> <span style='color:var(--red)'>++++++++++++++++++</span>    │
        18.9┤<span style='color:var(--green)'>·</span>                                 │28.9┤<span style='color:var(--red)'>++++</span>                          <span style='color:var(--red)'>++++</span>│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
            1.0     3.2      5.5     7.8   10.0     1.0     3.2      5.5     7.8   10.0 
        loss/mean           iter                loss/max            iter
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/loss.txt</span>
            ┌──────────────────────────────────────────────────────────────────────────┐
        56.3┤ <span style='color:var(--red)'>++</span> loss/max            <span style='color:var(--red)'>+</span>                                                 │
            │ <span style='color:var(--cyan)'>--</span> loss/min           <span style='color:var(--red)'>+</span> <span style='color:var(--red)'>+</span>                                                │
            │ <span style='color:var(--green)'>··</span> loss/mean         <span style='color:var(--red)'>+</span>   <span style='color:var(--red)'>+</span>                                               │
        49.3┤ ▞▞ loss             <span style='color:var(--red)'>+</span>     <span style='color:var(--red)'>++</span>                                             │
            │                    <span style='color:var(--red)'>+</span>        <span style='color:var(--red)'>+</span>                                            │
            │                   <span style='color:var(--red)'>+</span>          <span style='color:var(--red)'>+</span>                                           │
        42.3┤                  <span style='color:var(--red)'>+</span>            <span style='color:var(--red)'>+</span>▞▀▀▀▀▀▀▀▀▚▄▄▄▖                            │
            │                 <span style='color:var(--red)'>+</span>             ▞<span style='color:var(--green)'>·</span>         <span style='color:var(--red)'>+++</span>▝▀▀▀▚               <span style='color:var(--red)'>+</span>        │
            │        ▖<span style='color:var(--red)'>++++++++</span>       <span style='color:var(--green)'>······</span>▐<span style='color:var(--green)'>·</span><span style='color:var(--cyan)'>-</span><span style='color:var(--green)'>·········</span>        ▀▖           <span style='color:var(--red)'>++</span> <span style='color:var(--red)'>+</span>       │
        35.4┤       ▞▚<span style='color:var(--green)'>·</span>             <span style='color:var(--green)'>·</span>     ▗▘<span style='color:var(--cyan)'>-</span> <span style='color:var(--cyan)'>---------</span><span style='color:var(--green)'>········</span> ▝▚▖<span style='color:var(--red)'>+</span>     <span style='color:var(--red)'>+++</span>    <span style='color:var(--red)'>+</span>      │
            │      ▐<span style='color:var(--cyan)'>--</span>▚<span style='color:var(--green)'>··</span>         <span style='color:var(--green)'>··</span>     ▗▘<span style='color:var(--cyan)'>-</span>           <span style='color:var(--cyan)'>----</span>    <span style='color:var(--green)'>···</span>▝▄<span style='color:var(--red)'>+++++</span>     <span style='color:var(--green)'>·</span>  <span style='color:var(--red)'>++</span>    │
            │     ▗▘  <span style='color:var(--cyan)'>-</span>▌ <span style='color:var(--green)'>·</span>       <span style='color:var(--green)'>·</span>       ▞<span style='color:var(--cyan)'>-</span>                <span style='color:var(--cyan)'>----</span>    <span style='color:var(--green)'>·</span>▚▖<span style='color:var(--green)'>········</span> <span style='color:var(--green)'>··</span>  <span style='color:var(--red)'>+</span>   │
            │    <span style='color:var(--green)'>·</span>▌    ▝▖ <span style='color:var(--green)'>··</span>   <span style='color:var(--green)'>··</span>       ▞<span style='color:var(--cyan)'>-</span>                     <span style='color:var(--cyan)'>------</span>▝▚▄▖        <span style='color:var(--green)'>··</span> <span style='color:var(--red)'>+</span>  │
        28.4┤   <span style='color:var(--green)'>·</span>▞      ▝▖  <span style='color:var(--green)'>···</span>        ▐<span style='color:var(--cyan)'>-</span>                              <span style='color:var(--cyan)'>-</span>▝▀▚▄▖      <span style='color:var(--green)'>··</span><span style='color:var(--red)'>++</span>│
            │  <span style='color:var(--green)'>·</span>▗▘       ▚            ▗▘                                   <span style='color:var(--cyan)'>-</span>▝▀▀▄▄▖   <span style='color:var(--green)'>··</span>│
            │<span style='color:var(--red)'>+</span><span style='color:var(--green)'>·</span>▗▘         ▚          ▗▘                                        <span style='color:var(--cyan)'>--</span>▝▀▀▄▄▄│
        21.4┤<span style='color:var(--green)'>·</span> ▞           ▌        ▗▞                                                 │
            │<span style='color:var(--green)'>·</span>▞            ▝▖    ▗▄▀▘                                                  │
            │▗▘             ▝▖<span style='color:var(--cyan)'>-</span>▄▞▘                                                     │
        14.4┤▌               ▝▀                                                        │
            └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
            1.0               3.2                5.5               7.8             10.0 
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/loss_summary.txt</span>
                    loss/mean hist                           loss/max hist           
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        2.00┤                           ███████│2.00┤             ███████████          │
        1.67┤                           ███████│1.67┤             ███████████          │
        1.33┤                           ███████│1.33┤             ███████████          │
        1.00┤████   ███████   █████████████████│1.00┤███████   ██████████████      ████│
        0.67┤████   ███████   █████████████████│0.67┤███████   ██████████████      ████│
        0.33┤████   ███████   █████████████████│0.33┤███████   ██████████████      ████│
        0.00┤███    ██████    █████████████████│0.00┤███████   ██████████████      ████│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
        17.9    24.0     30.2    36.3   42.4    22.0    30.9     39.8    48.8   57.7 
                        loss/min hist                           loss/std hist           
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        2.00┤████                          ████│3.00┤████                              │
        1.67┤████                          ████│2.50┤████                              │
        1.33┤████                          ████│2.00┤██████████                        │
        1.00┤████   ██████████   ██████████████│1.50┤██████████                        │
            │████   ██████████   ██████████████│    │██████████                        │
        0.67┤████   ██████████   ██████████████│1.00┤██████████████      ████      ████│
        0.33┤████   ██████████   ██████████████│0.50┤██████████████      ████      ████│
        0.00┤███    ██████████   ██████████████│0.00┤█████████████       ████      ████│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
        13.3    20.1     26.9    33.7   40.5    -0.2     4.4      8.9    13.5   18.1 
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/loss_hist.txt</span>
        </pre>
        </div>

Use it to launch:

- Arbitrary command(s):

    ```bash
    ezpz launch hostname
    ```

- Arbitrary Python string:

    ```bash
    ezpz launch python3 -c 'import ezpz; ezpz.setup_torch()'
    ```

- One of the ready-to-go examples:

    ```bash
    ezpz launch python3 -m ezpz.test_dist --profile
    ezpz launch -n 8 -- python3 -m ezpz.examples.fsdp_tp --tp 4
    ```

- Your own distributed training script:

    ```bash
    ezpz launch -n 16 -ppn 8 -- python3 -m your_app.train --config configs/your_config.yaml
    ```

    to launch `your_app.train` across 16 processes, 8 per node.

[^schedulers]: By default, this will detect if we're running behind a job
    scheduler (e.g. PBS or Slurm).<br>
    If so, we automatically determine the specifics of the currently active
    job; explicitly, this will determine:

    1. The number of available nodes
    2. How many GPUs are present on each of these nodes
    3. How many GPUs we have _total_

    It will then use this information to automatically construct the
    appropriate {`mpiexec`, `srun`} command to launch, and finally, execute the
    launch cmd.


??? tip "Sequence Diagram"

    Two primary control paths drive `ezpz launch`: a scheduler-aware path used when
    running inside PBS/SLURM allocations, and a local fallback that shells out to
    `mpirun` when no scheduler metadata is available.

    ```mermaid
    sequenceDiagram
        autonumber
        actor User
        participant CLI as ezpz_launch
        participant Scheduler as PBS_or_Slurm
        participant MPI as mpirun_mpiexec
        participant App as User_application

        User->>CLI: ezpz launch <launch_flags> -- <cmd> <cmd_flags>
        CLI->>Scheduler: detect_scheduler()
        alt scheduler_detected
            Scheduler-->>CLI: scheduler_type, job_metadata
            CLI->>Scheduler: build_scheduler_command(cmd_to_launch)
            Scheduler-->>CLI: launch_cmd (mpiexec_or_srun)
            CLI->>MPI: run_command(launch_cmd)
            MPI->>App: start_ranks_and_execute
            App-->>MPI: return_codes
            MPI-->>CLI: aggregate_status
        else no_scheduler_detected
            Scheduler-->>CLI: unknown
            CLI->>MPI: mpirun -np 2 <cmd> <cmd_flags>
            MPI->>App: start_local_ranks
            App-->>MPI: return_codes
            MPI-->>CLI: aggregate_status
        end
        CLI-->>User: exit_code
    ```

## 📝 Ready-to-go Examples

- 📝 [`ezpz.examples.*`](../../examples/index.md): Scalable and _ready-to-go_!

    | Links                                                                                                                                                                                                                    | Example Module             | What it Does                                    |
    | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- | ----------------------------------------------- |
    | [:lucide-book:][ex-test-dist] · [:lucide-file-code:][api-test-dist] · [:lucide-github:][gh-test-dist]                                                                                                                    | `ezpz.examples.test_dist`  | Train MLP with DDP on MNIST                     |
    | [:lucide-book:][ex-fsdp] · [:lucide-file-code:][api-fsdp] · [:lucide-github:][gh-fsdp]                                                                                                                                   | `ezpz.examples.fsdp`       | Train CNN with FSDP on MNIST                    |
    | [:lucide-book:][ex-vit] · [:lucide-file-code:][api-vit] · [:lucide-github:][gh-vit]                            | `ezpz.examples.vit`        | Train ViT with FSDP on MNIST                    |
    | [:lucide-book:][ex-fsdp-tp] · [:lucide-file-code:][api-fsdp-tp] · [:lucide-github:][gh-fsdp-tp]                | `ezpz.examples.fsdp_tp`    | Train Transformer with FSDP + TP on HF Datasets |
    | [:lucide-book:][ex-diffusion] · [:lucide-file-code:][api-diffusion] · [:lucide-github:][gh-diffusion]          | `ezpz.examples.diffusion`  | Train Diffusion LLM with FSDP on HF Datasets    |
    | [:lucide-book:][ex-hf-trainer] · [:lucide-file-code:][api-hf-trainer] · [:lucide-github:][gh-hf-trainer] | `ezpz.examples.hf_trainer` | Train LLM with FSDP + HF Trainer on HF Datasets |


    - ??? tip "Running Examples"

            Any of the examples below can be launched with (sensible defaults if not
            specified):

            ```bash
            ezpz launch python3 -m ezpz.examples.fsdp
            ezpz launch python3 -m ezpz.examples.fsdp_tp
            # ...etc
            ezpz launch python3 -m ezpz.examples.hf_trainer
            ```

    - ??? tip "🤗 HF Integration"

            1. `ezpz.examples.`{[`fsdp_tp`][ex-fsdp-tp],
               [`diffusion`][ex-diffusion], [`hf_trainer`][ex-hf-trainer]} all
               support arbitrary 🤗 Hugging Face
               [datasets](https://huggingface.co/docs/datasets/index) e.g.:

                ```bash
                # use any --dataset from HF Datasets hub
                ezpz launch python3 -m ezpz.examples.fsdp_tp --dataset stanfordnlp/imdb
                ```

            1. [`ezpz.examples.hf_trainer`][ex-hf-trainer] supports arbitrary
               combinations of (compatible) `transformers.from_pretrained`
               models, and HF Datasets (with support for streaming!)

                ```bash
                ezpz launch python3 -m ezpz.examples.hf_trainer \
                    --streaming \
                    --dataset_name=eliplutchok/fineweb-small-sample \
                    --tokenizer_name meta-llama/Llama-3.2-1B \
                    --model_name_or_path meta-llama/Llama-3.2-1B \
                    --bf16=true
                    # ...etc.
                ```

    1. ??? example "`demo.py`"

            ```python title="demo.py"
            import ezpz

            # automatic device + backend setup for distributed PyTorch
            _ = ezpz.setup_torch()  # CUDA/NCCL, XPU/XCCL, {MPS, CPU}/GLOO, ...

            device = ezpz.get_torch_device() # {cuda, xpu, mps, cpu, ...}
            rank = ezpz.get_rank()
            world_size = ezpz.get_world_size()
            # ...etc

            if rank == 0:
                print(f"Hello from rank {rank} / {world_size} on {device}!")
            ```

            We can launch this script with:

            ```bash
            ezpz launch python3 demo.py
            ```

            ??? abstract "Output(s)"

                ??? success "MacBook Pro"

                    ```bash
                    # from MacBook Pro
                    $ ezpz launch python3 demo.py
                    [2026-01-08 07:22:31,989741][I][ezpz/launch:515:run] No active scheduler detected; falling back to local mpirun: mpirun -np 2 python3 /Users/samforeman/python/ezpz_demo.py
                    Using [2 / 2] available "mps" devices !!
                    Hello from rank 0 / 2 on mps!
                    ```

                ??? success "Aurora (2 nodes)"

                    ```bash
                    # from 2 nodes of Aurora:
                    #[aurora_frameworks-2025.2.0](foremans-aurora_frameworks-2025.2.0)[C v7.5.0-gcc][43s]
                    #[01/08/26,07:26:10][x4604c5s2b0n0][~]
                    ; ezpz launch python3 demo.py

                    [2026-01-08 07:26:19,723138][I][numexpr/utils:148:_init_num_threads] Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
                    [2026-01-08 07:26:19,725453][I][numexpr/utils:151:_init_num_threads] Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
                    [2026-01-08 07:26:19,725932][I][numexpr/utils:164:_init_num_threads] NumExpr defaulting to 16 threads.
                    [2026-01-08 07:26:20,290222][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2026-01-08-072620]----
                    [2026-01-08 07:26:21,566797][I][ezpz/launch:416:launch] Job ID: 8246832
                    [2026-01-08 07:26:21,567684][I][ezpz/launch:417:launch] nodelist: ['x4604c5s2b0n0', 'x4604c5s3b0n0']
                    [2026-01-08 07:26:21,568082][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                    [2026-01-08 07:26:21,568770][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
                    [2026-01-08 07:26:21,569557][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
                    [2026-01-08 07:26:21,569959][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
                    [2026-01-08 07:26:21,570821][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 demo.py
                    [2026-01-08 07:26:21,571548][I][ezpz/launch:433:launch] Took: 2.11 seconds to build command.
                    [2026-01-08 07:26:21,571918][I][ezpz/launch:436:launch] Executing:
                    mpiexec
                    --envall
                    --np=24
                    --ppn=12
                    --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                    --no-vni
                    --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
                    python3
                    demo.py
                    [2026-01-08 07:26:21,573262][I][ezpz/launch:220:get_aurora_filters] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG
                    [2026-01-08 07:26:21,573781][I][ezpz/launch:443:launch] Execution started @ 2026-01-08-072621...
                    [2026-01-08 07:26:21,574195][I][ezpz/launch:138:run_command] Caught 24 filters
                    [2026-01-08 07:26:21,574532][I][ezpz/launch:139:run_command] Running command:
                    mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 demo.py
                    cpubind:list x4604c5s3b0n0 pid 131587 rank 12 0: mask 0x1c
                    cpubind:list x4604c5s3b0n0 pid 131588 rank 13 1: mask 0x1c00
                    cpubind:list x4604c5s3b0n0 pid 131589 rank 14 2: mask 0x1c0000
                    cpubind:list x4604c5s3b0n0 pid 131590 rank 15 3: mask 0x1c000000
                    cpubind:list x4604c5s3b0n0 pid 131591 rank 16 4: mask 0x1c00000000
                    cpubind:list x4604c5s3b0n0 pid 131592 rank 17 5: mask 0x1c0000000000
                    cpubind:list x4604c5s3b0n0 pid 131593 rank 18 6: mask 0x1c0000000000000
                    cpubind:list x4604c5s3b0n0 pid 131594 rank 19 7: mask 0x1c000000000000000
                    cpubind:list x4604c5s3b0n0 pid 131595 rank 20 8: mask 0x1c00000000000000000
                    cpubind:list x4604c5s3b0n0 pid 131596 rank 21 9: mask 0x1c0000000000000000000
                    cpubind:list x4604c5s3b0n0 pid 131597 rank 22 10: mask 0x1c000000000000000000000
                    cpubind:list x4604c5s3b0n0 pid 131598 rank 23 11: mask 0x1c00000000000000000000000
                    cpubind:list x4604c5s2b0n0 pid 121225 rank 0 0: mask 0x1c
                    cpubind:list x4604c5s2b0n0 pid 121226 rank 1 1: mask 0x1c00
                    cpubind:list x4604c5s2b0n0 pid 121227 rank 2 2: mask 0x1c0000
                    cpubind:list x4604c5s2b0n0 pid 121228 rank 3 3: mask 0x1c000000
                    cpubind:list x4604c5s2b0n0 pid 121229 rank 4 4: mask 0x1c00000000
                    cpubind:list x4604c5s2b0n0 pid 121230 rank 5 5: mask 0x1c0000000000
                    cpubind:list x4604c5s2b0n0 pid 121231 rank 6 6: mask 0x1c0000000000000
                    cpubind:list x4604c5s2b0n0 pid 121232 rank 7 7: mask 0x1c000000000000000
                    cpubind:list x4604c5s2b0n0 pid 121233 rank 8 8: mask 0x1c00000000000000000
                    cpubind:list x4604c5s2b0n0 pid 121234 rank 9 9: mask 0x1c0000000000000000000
                    cpubind:list x4604c5s2b0n0 pid 121235 rank 10 10: mask 0x1c000000000000000000000
                    cpubind:list x4604c5s2b0n0 pid 121236 rank 11 11: mask 0x1c00000000000000000000000
                    Using [24 / 24] available "xpu" devices !!
                    Hello from rank 0 / 24 on xpu!
                    [2026-01-08 07:26:33,060432][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2026-01-08-072633]----
                    [2026-01-08 07:26:33,061512][I][ezpz/launch:448:launch] Execution finished with 0.
                    [2026-01-08 07:26:33,062045][I][ezpz/launch:449:launch] Executing finished in 11.49 seconds.
                    [2026-01-08 07:26:33,062531][I][ezpz/launch:450:launch] Took 11.49 seconds to run. Exiting.
                    took: 22s
                    ```

    - ??? example "Simple Example"

            ```bash
            ezpz launch python3 -c 'import ezpz; print(ezpz.setup_torch())'
            ```

            ??? success "Output"

                ??? example "Macbook Pro"

                    ```bash
                    #[01/08/26 @ 14:56:50][~/v/s/ezpz][dev][$✘!?] [4s]
                    ; ezpz launch python3 -c 'import ezpz; print(ezpz.setup_torch())'
                    [2026-01-08 14:56:54,307030][I][ezpz/launch:515:run] No active scheduler detected; falling back to local mpirun: mpirun -np 2 python3 -c 'import ezpz; print(ezpz.setup_torch())'
                    Using [2 / 2] available "mps" devices !!
                    0
                    1
                    [2025-12-23-162222] Execution time: 4s sec
                    ```

                ??? example "Aurora (2 Nodes)"

                    ```bash
                    #[aurora_frameworks-2025.2.0](torchtitan-aurora_frameworks-2025.2.0)[1m9s]
                    #[01/08/26,14:56:42][x4418c6s1b0n0][/f/d/f/p/p/torchtitan][main][?]
                    ; ezpz launch python3 -c 'import ezpz; print(ezpz.setup_torch())'


                    [2026-01-08 14:58:01,994729][I][numexpr/utils:148:_init_num_threads] Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
                    [2026-01-08 14:58:01,997067][I][numexpr/utils:151:_init_num_threads] Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
                    [2026-01-08 14:58:01,997545][I][numexpr/utils:164:_init_num_threads] NumExpr defaulting to 16 threads.
                    [2026-01-08 14:58:02,465850][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2026-01-08-145802]----
                    [2026-01-08 14:58:04,765720][I][ezpz/launch:416:launch] Job ID: 8247203
                    [2026-01-08 14:58:04,766527][I][ezpz/launch:417:launch] nodelist: ['x4418c6s1b0n0', 'x4717c0s6b0n0']
                    [2026-01-08 14:58:04,766930][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                    [2026-01-08 14:58:04,767616][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
                    [2026-01-08 14:58:04,768399][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
                    [2026-01-08 14:58:04,768802][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
                    [2026-01-08 14:58:04,769517][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 -c 'import ezpz; print(ezpz.setup_torch())'
                    [2026-01-08 14:58:04,770278][I][ezpz/launch:433:launch] Took: 3.01 seconds to build command.
                    [2026-01-08 14:58:04,770660][I][ezpz/launch:436:launch] Executing:
                    mpiexec
                    --envall
                    --np=24
                    --ppn=12
                    --hostfile=/var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                    --no-vni
                    --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
                    python3
                    -c
                    import ezpz; print(ezpz.setup_torch())
                    [2026-01-08 14:58:04,772125][I][ezpz/launch:220:get_aurora_filters] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG
                    [2026-01-08 14:58:04,772651][I][ezpz/launch:443:launch] Execution started @ 2026-01-08-145804...
                    [2026-01-08 14:58:04,773070][I][ezpz/launch:138:run_command] Caught 24 filters
                    [2026-01-08 14:58:04,773429][I][ezpz/launch:139:run_command] Running command:
                    mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 -c 'import ezpz; print(ezpz.setup_torch())'
                    cpubind:list x4717c0s6b0n0 pid 118589 rank 12 0: mask 0x1c
                    cpubind:list x4717c0s6b0n0 pid 118590 rank 13 1: mask 0x1c00
                    cpubind:list x4717c0s6b0n0 pid 118591 rank 14 2: mask 0x1c0000
                    cpubind:list x4717c0s6b0n0 pid 118592 rank 15 3: mask 0x1c000000
                    cpubind:list x4717c0s6b0n0 pid 118593 rank 16 4: mask 0x1c00000000
                    cpubind:list x4717c0s6b0n0 pid 118594 rank 17 5: mask 0x1c0000000000
                    cpubind:list x4717c0s6b0n0 pid 118595 rank 18 6: mask 0x1c0000000000000
                    cpubind:list x4717c0s6b0n0 pid 118596 rank 19 7: mask 0x1c000000000000000
                    cpubind:list x4717c0s6b0n0 pid 118597 rank 20 8: mask 0x1c00000000000000000
                    cpubind:list x4717c0s6b0n0 pid 118598 rank 21 9: mask 0x1c0000000000000000000
                    cpubind:list x4717c0s6b0n0 pid 118599 rank 22 10: mask 0x1c000000000000000000000
                    cpubind:list x4717c0s6b0n0 pid 118600 rank 23 11: mask 0x1c00000000000000000000000
                    cpubind:list x4418c6s1b0n0 pid 66450 rank 0 0: mask 0x1c
                    cpubind:list x4418c6s1b0n0 pid 66451 rank 1 1: mask 0x1c00
                    cpubind:list x4418c6s1b0n0 pid 66452 rank 2 2: mask 0x1c0000
                    cpubind:list x4418c6s1b0n0 pid 66453 rank 3 3: mask 0x1c000000
                    cpubind:list x4418c6s1b0n0 pid 66454 rank 4 4: mask 0x1c00000000
                    cpubind:list x4418c6s1b0n0 pid 66455 rank 5 5: mask 0x1c0000000000
                    cpubind:list x4418c6s1b0n0 pid 66456 rank 6 6: mask 0x1c0000000000000
                    cpubind:list x4418c6s1b0n0 pid 66457 rank 7 7: mask 0x1c000000000000000
                    cpubind:list x4418c6s1b0n0 pid 66458 rank 8 8: mask 0x1c00000000000000000
                    cpubind:list x4418c6s1b0n0 pid 66459 rank 9 9: mask 0x1c0000000000000000000
                    cpubind:list x4418c6s1b0n0 pid 66460 rank 10 10: mask 0x1c000000000000000000000
                    cpubind:list x4418c6s1b0n0 pid 66461 rank 11 11: mask 0x1c00000000000000000000000
                    Using [24 / 24] available "xpu" devices !!
                    8
                    10
                    0
                    4
                    3
                    5
                    7
                    11
                    6
                    1
                    9
                    2
                    14
                    15
                    12
                    13
                    16
                    17
                    19
                    22
                    20
                    23
                    18
                    21
                    [2026-01-08 14:58:14,252433][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2026-01-08-145814]----
                    [2026-01-08 14:58:14,253726][I][ezpz/launch:448:launch] Execution finished with 0.
                    [2026-01-08 14:58:14,254184][I][ezpz/launch:449:launch] Executing finished in 9.48 seconds.
                    [2026-01-08 14:58:14,254555][I][ezpz/launch:450:launch] Took 9.48 seconds to run. Exiting.
                    took: 18s
                    ```

  [ex-test-dist]: ../../examples/test-dist.md "Example"
  [api-test-dist]: ../../python/Code-Reference/test_dist.md "API Reference"
  [gh-test-dist]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py "GitHub Source"
  [ex-fsdp]: ../../examples/fsdp.md "Example"
  [api-fsdp]: ../../python/Code-Reference/examples/fsdp.md "API Reference"
  [gh-fsdp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py "GitHub Source"
  [ex-vit]: ../../examples/vit.md "Example"
  [api-vit]: ../../python/Code-Reference/examples/vit.md "API Reference"
  [gh-vit]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py "GitHub Source"
  [ex-fsdp-tp]: ../../examples/fsdp-tp.md "Example"
  [api-fsdp-tp]: ../../python/Code-Reference/examples/fsdp_tp.md "API Reference"
  [gh-fsdp-tp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py "GitHub Source"
  [ex-diffusion]: ../../examples/diffusion.md "Example"
  [api-diffusion]: ../../python/Code-Reference/examples/diffusion.md "API Reference"
  [gh-diffusion]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py "GitHub Source"
  [ex-hf-trainer]: ../../examples/hf-trainer/index.md "Example"
  [api-hf-trainer]: ../../python/Code-Reference/examples/hf_trainer.md "API Reference"
  [gh-hf-trainer]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py "GitHub Source"

[^distributed-history]: The `ezpz.History` class automatically computes
    distributed statistics (min, max, mean, std. dev) across ranks for all
    recorded metrics.  
    **NOTE**: This is automatically disabled when
    `ezpz.get_world_size() >= 384` (e.g. >= {32, 96} {Aurora, Polaris} nodes)
    due to the additional overhead introduced (but can be manually enabled, if
    desired).
