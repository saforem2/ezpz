# Drafts: ALCF / NERSC support tickets

**Both unsent.** Filing is the user's call — these are drafts so it is
copy-paste rather than rewriting from scratch.

---

## 1. ALCF (Polaris): every conda module fails to load

**Severity:** blocks all conda users on Polaris, not just this project.

**Summary.** Every `conda` module on Polaris fails with
`Lmod has detected the following error: The following module(s) are
unknown`, because the modulefiles require two modules the system no
longer provides.

**Reproduce** (login node, 2026-08-26):

```bash
module use /soft/modulefiles
module load conda/2025-09-25
```

Result:

```
The following module(s) are unknown: "gcc-native/14.2"
                                     "cray-hdf5-parallel/1.14.3.5"
```

What the system actually has now:

| modulefile requires | system provides |
|---|---|
| `gcc-native/14.2` | `gcc-native/14` |
| `cray-hdf5-parallel/1.14.3.5` | `cray-hdf5-parallel/1.14.3.9` |

**Affects every conda module**, not one: `conda/2024-04-29`,
`conda/2025-09-25`, `conda/2025-09-26-aws-nccl-1.6.0`,
`conda/2025-09-26-aws-nccl-1.9.1`.

**Downstream symptom.** `conda` never lands on `PATH`, so anything
depending on it fails much later with a misleading message — in our
case `CONDA_PREFIX still not set`, several layers from the real cause.

**Ask.** Update the conda modulefiles to the versions currently
installed (or restore the pinned ones).

**Workaround we used.** Build a standalone venv on Cray's python,
`/opt/cray/pe/python/3.12.12` — it also has a proper `RUNPATH` and a
working `mpi4py`, which a `uv`-managed CPython does not.

---

## 2. NERSC (Perlmutter): aws-ofi-nccl deadlocks a ~4 MiB reduce-scatter

**Status: ready to file.** Isolated 2026-09-15 to the network plugin.

**Summary.** A PyTorch reduce-scatter of ~4 MiB deadlocks deterministically
on Perlmutter's `aws-ofi-nccl` / Slingshot (`cxi`) datapath. The identical
job on the identical nodes completes when NCCL is forced onto TCP
(`NCCL_NET=Socket`). This is a same-machine, same-allocation controlled
swap: only the network layer changes.

**Configuration.** 2 x (4 x A100-40GB), `world_size=8`, torch
`2.13.0+cu130`, NCCL **2.29.7**, `aws-ofi-nccl 1.6.0`, provider `cxi`
(4 NICs), `TORCH_DDP_TIMEOUT=180-300`.

**Evidence** (jobs 58365643, 58368557, 58369713):

| configuration | runs | result |
|---|---|---|
| default (`aws-ofi-nccl`) | **6/6** | **HANG**, always `NumelIn=1055232` |
| `NCCL_NET=Socket` | **3/3** | trains, ~7x slower (1.38 vs 0.20 s/step) |
| `NCCL_PROTO=Simple` | 1/1 | HANG, byte-identical |
| `NCCL_ALGO=Ring` | 1/1 | HANG, byte-identical |

The six hangs include two clean baselines plus every diagnostic variant.
All stall on a **byte-identical** buffer, so NCCL's protocol and algorithm
selection are both excluded — the defect is beneath them. The socket runs
are genuine TCP, confirmed by the ~7x slowdown matching the documented
inter-node fallback penalty.

**The stalling collective.** All eight ranks report the same state:

```
WorkNCCL(SeqNum=18, OpType=_REDUCE_SCATTER_BASE,
         NumelIn=1055232, NumelOut=131904, Timeout(ms)=300000)
```

`TORCH_NCCL_DESYNC_DEBUG=1` confirms no rank diverged:

```
[0,1,2,3,4,5,6,7] joined but didn't finish collective #39
```

Every rank entered. This is not an application-side participation bug.
The payload is fp32: `1055232 x 4 B = 4.025 MiB` in, `515 KiB` per shard.
The buffer size is invariant across runs while the sequence number shifts
under instrumentation (18 -> 20), so the trigger is the **message**, not
its position in the stream.

**Two defects we would like NERSC to look at.**

**(a) The NCCL a job loads is not the NCCL it asks for.** Our batch
scripts run `module load nccl/2.24.3`, but the NERSC PyTorch 2.13 install
bundles its own `libnccl.so.2` at **2.29.7** under
`site-packages/nvidia/nccl/lib/`, which wins via RPATH. The module load is
inert. So `aws-ofi-nccl 1.6.0` is running against an NCCL five minor
versions newer than the pairing we assume was validated. If that pairing
is unsupported, this alone may be the whole bug — and every user doing
`module load nccl/...` alongside the site PyTorch is silently in the same
position.

**(b) No fallback path, so it hangs instead of failing.** Every rank on
both nodes emits at init:

```
misc/ibvwrap.cc:173 NCCL WARN lib wrapper not initialized.
```

Benign on its face, since `cxi` is the transport. But it means a stalled
OFI operation has nothing to fail over to. The last `NCCL INFO` lines
before the stall are `[Service thread] Connection closed by localRank N`
— teardown, not data movement.

**Ruled out** (each tested on Perlmutter, not assumed): PyTorch/FSDP2
itself (all ranks join; and the same code trains on three other
machines), torch version (byte-identical on 2.11 and 2.13), NCCL protocol
and algorithm selection, collective ordering, the frozen-unit
all-gather/reduce-scatter asymmetry, and every alignment threshold we
could construct.

**A standalone reduce-scatter of the same size does NOT reproduce it.**
`experiments/common/reduce_scatter_size_sweep.py` calls
`dist.reduce_scatter_tensor` directly, with no FSDP2/LoRA/ezpz, across
0.5-12.8 MiB. Run on Perlmutter at the identical geometry (ws=8, 2x4
A100, torch 2.13, default `aws-ofi-nccl`/`cxi`), **every size completed
in under 0.1 s with zero watchdog timeouts**, including `NumelIn=1055232`
-- the exact buffer the LoRA job deadlocks on.

So the payload size alone is not the trigger. Something about how FSDP2
drives the collective matters: concurrency with in-flight all-gathers,
two CUDA streams sharing one communicator, or the surrounding sequence.
We report this rather than omit it, because it bounds where to look.

The same sweep passes on Polaris (A100/NCCL), Sunspot and Aurora
(PVC/XCCL) -- a useful cross-machine control, but not a reproducer.

**What still stands:** on the same nodes, in the same allocation, the
LoRA job hangs **6/6** on the default path and trains **3/3** under
`NCCL_NET=Socket`. The transport swap is the strongest evidence and is
unaffected by the negative result above.

**Reproducer — use this one.**
`experiments/perlmutter/lora_239_transport.sbatch` in
`saforem2/ezpz` — one debug-QOS job, four arms, prints a verdict per arm.

**Caveat, stated honestly.** Our cross-machine control (Polaris, also
A100 + NCCL, trains clean) loads **no** NCCL network plugin, so it varied
site *and* plugin rather than site alone. It is therefore a second
plugin-free configuration that works, not an independent site control.
The same-machine `NCCL_NET=Socket` swap above is the stronger evidence
and does not depend on it.

## 3. ALCF (Aurora): `frameworks/2026.1.0` — `import torch` fails

**Status: ready to file.** Bug 1 only; see the caveat at the end.

**Summary.** On `frameworks/2026.1.0`, a bare `import torch` raises
`OSError: libglog.so.0: cannot open shared object file`. This is a
packaging defect in the module, not a user environment problem, and it is
a **regression** — `frameworks/2025.3.1` is unaffected.

**Root cause.** `libtorchcomms.so`'s RUNPATH contains **no entry pointing
at the install prefix** where `libglog.so.0` actually ships. It lists
build-time paths under `/lus/tegu` — a Sunspot filesystem, not mounted on
Aurora — and two **Windows** library directories:

```
/opt/aurora/26.181.0/oneapi/mkl/latest/lib/intel64_win
/opt/aurora/26.181.0/oneapi/mkl/latest/lib/win-x64
```

torch 2.13 imports `torchcomms` unconditionally
(`distributed_c10d.py:151`), so a dangling RUNPATH is fatal at
`import torch`. `frameworks/2025.3.1` ships an identically mis-linked
`libtorchcomms.so` but never imports it, which is why only 2026.1.0
breaks.

**Login and compute run different release trees.** This is what makes the
bug hard to see: the login nodes load `/opt/aurora/26.26.0/...-2025.3.1`
(torch 2.10, unaffected), while `next-eval` compute nodes load
`/opt/aurora/26.181.0/...-2026.1.0` (torch 2.13, broken). The `26.181.0`
path does not even resolve from a login node, so a login-node check
reports the module as absent rather than broken.

**Workaround** (re-verified 2026-09-18 on `next-eval`, 2 nodes, ws=24 --
`import torch` succeeds and a full collective sweep completes):

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
```

**Reproducer.** `experiments/aurora/frameworks_2026_1_0_bugs.pbs` in
`saforem2/ezpz` — a standalone PBS script needing no checkout. Full
verified output, including the RUNPATH dump, is in
`docs/notes/aurora-frameworks-2026.1.0-bugs.md`.

**Do NOT include the wandb symptom in this ticket.** We initially took
`import wandb` failing with an `AttributeError` to be a second module bug.
It is not ALCF's: wandb is simply not installed, and the confusing symptom
is a stray `~/wandb` run-output directory being picked up as an implicit
namespace package. Filing it would waste ALCF's time.
