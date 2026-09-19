# Aurora `frameworks/2026.1.0`: verified bug report

**One of these is an ALCF defect worth a ticket. The other is not — read
the verdict table before filing.**

## Reproduce it yourself — one line, no checkout needed

```bash
ssh aurora 'curl -fsSL https://raw.githubusercontent.com/saforem2/ezpz/main/experiments/aurora/frameworks_2026_1_0_bugs.pbs | /opt/pbs/bin/qsub'
```

> **Before #240 merges** the file is not on `main` yet — swap `main` for
> `fix/239-frozen-unit-reshard` in that URL, or it 404s.

`qsub` reads the script from stdin and parses the `#PBS` directives out of
the piped text, so there is no temp file and no `bash -c` wrapper. Two
things that will break it:

- `/opt/pbs/bin/qsub` must be the **full path** — `qsub` is not on `$PATH`
  over a plain `ssh` command.
- The `curl` runs on the **login node**, which has internet. Compute nodes
  do not, but they never need it: PBS has already captured the script.

Output lands in `~/fw2026-bugs.o<jobid>`. Script:
[`experiments/aurora/frameworks_2026_1_0_bugs.pbs`](../../experiments/aurora/frameworks_2026_1_0_bugs.pbs).

## Verdict

| | status | file a ticket? |
|---|---|---|
| **Bug 1** — `import torch` → `OSError: libglog.so.0` | genuine ALCF packaging defect, 2026.1.0-only | **yes** |
| **Bug 2** — `import wandb` → `AttributeError` | wandb is simply not installed; the confusing symptom is a stray `~/wandb` shadowing it | **no** |

## Bug 1 — the case, in one block

`libtorchcomms.so`'s RUNPATH contains **no entry pointing at the install
prefix** where `libglog.so.0` actually ships. It lists build-time paths
under `/lus/tegu` — a Sunspot filesystem, not mounted on Aurora — and,
oddly, two **Windows** library directories:

```
/opt/aurora/26.181.0/oneapi/mkl/latest/lib/intel64_win
/opt/aurora/26.181.0/oneapi/mkl/latest/lib/win-x64
```

torch 2.13 imports `torchcomms` unconditionally (`distributed_c10d.py:151`),
so this is fatal at `import torch`. `frameworks/2025.3.1` ships an
identically mis-linked `libtorchcomms.so` but never imports it — which
makes 2026.1.0 a regression rather than a long-standing quirk.

**Workaround:** `export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"`

## Bug 2 — why this is NOT ALCF's bug

`wandb` is absent from 2026.1.0 (2025.3.1 has it, 35 entries +
`wandb-0.24.2.dist-info`). That alone would give a clean
`ModuleNotFoundError`. The confusing `AttributeError` comes from somewhere
else: PBS starts jobs with `cwd=$HOME`, cwd is on `sys.path`, and the
**wandb run-output directory** `~/wandb` (119 entries: `run-*/`,
`latest-run`, `debug.log`) becomes an implicit namespace package.

The script proves it by re-importing from `/tmp`, which yields the honest
`ModuleNotFoundError`. Note this would shadow a *working* install too, so
`pip install --user wandb` is not a reliable fix on its own — you also must
not run from a directory containing a `wandb/` folder.

## Verified output

Job `8792487`, node `x4413c2s2b0n0`, `queue=validation`, `Exit_status=0`,
walltime 8s — submitted with the exact one-liner above.

```text
==================================================================
 node            : x4413c2s2b0n0
 frameworks      : frameworks/2026.1.0
 CONDA_PREFIX    : /opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
 python3         : /opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python3
==================================================================

################ BUG 1: torch / libglog ################

--- 1a. import torch WITHOUT the workaround (expect: OSError libglog.so.0)
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: libglog.so.0: cannot open shared object file: No such file or directory
    rc=1

--- 1b. the library DOES exist, it is just unreachable:
lrwxrwxrwx 1 root root     16 Aug 14 23:58 /opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/libglog.so -> libglog.so.0.4.0
lrwxrwxrwx 1 root root     16 Aug 14 23:58 /opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/libglog.so.0 -> libglog.so.0.4.0
-rwxr-xr-x 1 root root 195368 Apr 17  2020 /opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/libglog.so.0.4.0

--- 1c. the dangling RUNPATH that causes it:
  RUNPATH              /opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/gcc-14.3.0-siitp7a/lib64:/lus/tegu/projects/datasets/software/26.181.0/wheelforge/envs/conda_envs/triton_3.7.1_torchcomms_0.3.1-rc1_vllm_0.25.1_nre_pt_2.13.0_rel_one_2026.1.0_np_2.3.5_python_3.12.12/lib:/opt/aurora/26.181.0/oneapi/mkl/latest/lib/intel64_win:/opt/aurora/26.181.0/oneapi/mkl/latest/lib/win-x64:/lus/tegu/projects/datasets/software/26.181.0/wheelforge/envs/conda_envs/triton_3.7.1_torchcomms_0.3.1-rc1_vllm_0.25.1_nre_pt_2.13.0_rel_one_2026.1.0_np_2.3.5_python_3.12.12/lib/python3.12/site-packages/torch/lib
    -> missing libs per ldd:
	libtorch.so => not found
	libc10.so => not found
	libc10_xpu.so => not found
	libtorch_xpu.so => not found

--- 1d. WITH the workaround (expect: torch imports)
torch 2.13.0a0+gitcf30153

################ BUG 2: empty wandb namespace package ################

--- 2a. import wandb, user-site DISABLED (expect: AttributeError)
  File "<string>", line 1, in <module>
AttributeError: module 'wandb' has no attribute '__version__'
    rc=1   (python3 -s == ignore ~/.local)

--- 2a2. same import WITH user-site (shows whether you already fixed it)
wandb 0.29.0

--- 2b. why: the directory is EMPTY and has no dist-info
    site-packages/wandb entries : 0
    dist-info                   : NONE (pip believes nothing is installed)

--- 2c. contrast with 2025.3.1, which is fine:
    2025.3.1 wandb entries      : 35
    2025.3.1 dist-info          : wandb-0.24.2.dist-info

--- 2c2. IS THE REAL SHADOW A STRAY ./wandb IN CWD? (PBS cwd = $HOME)
    cwd: /home/foremans
    FOUND /home/foremans/wandb  -- 119 entries
    looks like wandb RUN OUTPUT, not a package:
      debug-internal.log
      debug.log
      latest-run
    -> cwd is on sys.path, so this becomes a namespace package
    -> proof: same import from a clean dir:
ModuleNotFoundError: No module named 'wandb'

--- 2d. it imports as a namespace package (no __file__ of its own):
    __file__: None
    __path__: ['/home/foremans/wandb']
    n attrs : 0

==================================================================
 SUMMARY
==================================================================
 BUG 1 torch/libglog reproduced : YES
 BUG 2 empty wandb reproduced   : YES

 Workarounds:
   export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
   python3 -m pip install --user wandb

 NOTE: BUG 2 is probed with `python3 -s` (user site-packages disabled)
 so a previously-installed ~/.local wandb cannot mask it. Step 2a2 shows
 the un-suppressed import, i.e. what your own environment actually sees.
==================================================================
```
