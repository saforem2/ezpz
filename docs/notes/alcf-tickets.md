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

## 2. NERSC (Perlmutter): FSDP2 + LoRA deadlock, Perlmutter-only

**Summary.** A PyTorch FSDP2 training job deadlocks deterministically in
the first backward pass on Perlmutter, and the identical configuration
runs clean on three other machines — including another A100 + NCCL
system at the same world size. That points at something in Perlmutter's
software stack rather than at PyTorch or NCCL as shipped.

**Configuration.** 2 nodes x 4 A100-40GB, `world_size=8`,
torch `2.13.0+cu130`, NCCL 2.29.7, `module load nccl/2.24.3`
(AWS-libfabric plugin), `TORCH_DDP_TIMEOUT=300`.

**Symptom.** All eight ranks report byte-identical watchdog state:

```
WorkNCCL(SeqNum=18, OpType=_REDUCE_SCATTER_BASE,
         NumelIn=419840, NumelOut=52480, Timeout(ms)=300000)

Timeout at collective: _ALLGATHER_BASE, #39
  [0,1,2,3,4,5,6,7] joined but didn't finish collective #39

PG status: last enqueued work: 39,
           last started work: 19 (_ALLGATHER_BASE),
           last completed work: 17
```

Note `last completed: 17` -> `last started: 19` — **work #18 is
skipped**, and #19 is an all-gather. The stream ran ahead of the
reduce-scatter it is now blocked on. Reproduced **6/6** across four
jobs (`57601590` x3, `57602201`, `57604574`).

**Why we believe it is Perlmutter-specific.** Same code, same model,
same `world_size=8`:

| | Perlmutter<br>A100/NCCL/2.13 | Polaris<br>A100/NCCL/2.13 | Sunspot<br>PVC/xccl/2.13 | Aurora<br>PVC/xccl/2.10 |
|---|---|---|---|---|
| hang? | **yes, 6/6** | no | no | no |

Polaris matches Perlmutter on accelerator, collectives, world size and
torch minor, varying only the site — and it trains clean.

**Also ruled out on Perlmutter** (each tested, not assumed): the
LoRA-specific hypothesis, payload size, per-rank collective ordering,
the frozen-unit all-gather/reduce-scatter asymmetry, a 256 KiB protocol
threshold, and NCCL protocol selection (`NCCL_PROTO=Simple` and `LL128`
both still hang).

**Remaining suspects, all Perlmutter-side.** NCCL 2.29.7, the
`nccl/2.24.3` AWS-libfabric plugin, Slingshot, or the CUDA 13.0 / cu130
pairing.

**Questions.**

1. Are there known issues with NCCL 2.29.7 or the current libfabric
   plugin around reduce-scatter completion under FSDP2?
2. Is a different NCCL or plugin version recommended for torch 2.13 +
   cu130 on Perlmutter?
3. Can you reproduce with the standalone script we can supply (torch +
   torchrun only, no site dependencies)?

**Detail.** Full write-up, including every refuted hypothesis, is in
`docs/guides/lora-fsdp-deadlock.md` in this repo.
