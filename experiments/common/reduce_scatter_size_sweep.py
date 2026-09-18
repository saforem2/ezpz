"""Standalone reduce-scatter size sweep. No FSDP2, no LoRA, no ezpz.

ezpz #239: a ~4 MiB reduce-scatter deadlocks on Perlmutter's aws-ofi-nccl
/ cxi path and completes over TCP. The evidence so far is LoRA-shaped,
which makes it a poor reproducer for a site ticket. This isolates the
claim to its essentials: does a bare reduce_scatter_tensor of size N hang?

Sweeps sizes bracketing the observed window [1.602, 4.025] MiB known to
hang and the 4.137 MiB known to train, so the output either draws the
boundary or refutes the window framing outright.

Each size gets its own timeout: a hang is reported and the sweep moves
on, so ONE job characterises the whole curve.
"""
import os, sys, time
import torch, torch.distributed as dist

def _derive_rank_env():
    """Fill RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR before init_process_group.

    Three launchers, three conventions, and one of them supplies nothing:

    * srun exports SLURM_PROCID/SLURM_NTASKS.
    * Some mpiexec builds export PMI_RANK/PMI_SIZE.
    * Aurora/Sunspot PALS mpiexec exports NEITHER -- a probe job on
      Sunspot found no rank-bearing variable in the environment at all,
      which is why an env-only implementation failed there with
      "unset: ['RANK','WORLD_SIZE']" on every one of 24 ranks.

    So fall back to mpi4py, which asks the MPI runtime directly and is how
    ezpz itself resolves rank on these machines. Broadcast rank 0's
    hostname for MASTER_ADDR so the rendezvous target is agreed rather
    than guessed.
    """
    env = os.environ
    pairs = (
        ("RANK", ("PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID")),
        ("WORLD_SIZE", ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS")),
        ("LOCAL_RANK", ("PALS_LOCAL_RANKID", "MPI_LOCALRANKID",
                        "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID")),
    )
    for want, cands in pairs:
        if env.get(want):
            continue
        for c in cands:
            if env.get(c) is not None:
                env[want] = env[c]
                break

    if not (env.get("RANK") and env.get("WORLD_SIZE")):
        from mpi4py import MPI                      # noqa: PLC0415
        comm = MPI.COMM_WORLD
        env["RANK"] = str(comm.Get_rank())
        env["WORLD_SIZE"] = str(comm.Get_size())
        if not env.get("LOCAL_RANK"):
            # Rank order within this host = local rank.
            host = MPI.Get_processor_name()
            hosts = comm.allgather(host)
            env["LOCAL_RANK"] = str(
                [i for i, h in enumerate(hosts) if h == host].index(comm.Get_rank())
            )
        if not env.get("MASTER_ADDR"):
            env["MASTER_ADDR"] = comm.bcast(MPI.Get_processor_name(), root=0)
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("MASTER_PORT", "29577")


def main():
    _derive_rank_env()
    # torch 2.13 renamed this; support both without a FutureWarning.
    global _rs
    _rs = getattr(dist, "reduce_scatter_single", dist.reduce_scatter_tensor)
    missing = [v for v in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
               if not os.environ.get(v)]
    if missing:
        raise SystemExit(f"FATAL: cannot rendezvous, unset: {missing}. "
                         f"seen: PMI_RANK={os.environ.get('PMI_RANK')} "
                         f"SLURM_PROCID={os.environ.get('SLURM_PROCID')}")
    dist.init_process_group(backend=sys.argv[1] if len(sys.argv) > 1 else "nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", 0))
    # Pick the accelerator that is actually USABLE. Testing xpu via a bare
    # hasattr is not enough: a CUDA build exposes torch.xpu but raises
    # "Torch not compiled with XPU enabled" on first use, so a CPU-only
    # login node fell through to xpu and died before any collective ran.
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{lr}")
        torch.cuda.set_device(dev)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        dev = torch.device(f"xpu:{lr}")
        torch.xpu.set_device(dev)
    else:
        dev = torch.device("cpu")  # gloo smoke test

    # numels chosen so numel*4 brackets the window; all divisible by ws.
    NUMELS = [
        131072,    # 0.5 MiB  - below window
        262144,    # 1.0 MiB  - below
        419840,    #          - the EXACT r8 payload that hangs
        524288,    # 2.0 MiB  - inside
        839680,    #          - the EXACT r16 payload (3.203 MiB), hangs
        1055232,   #          - the EXACT r17 payload that hangs
        1084416,   #          - the EXACT r18 payload that TRAINS
        1179648,   # 4.5 MiB  - above
        2097152,   # 8.0 MiB  - well above
        3358720,   #          - the EXACT r64 payload that trains
    ]
    if rank == 0:
        print(f"# backend={dist.get_backend()} ws={ws} dev={dev.type} "
              f"torch={torch.__version__}", flush=True)
        print("# numel\tMiB_in\tverdict\tsecs", flush=True)

    for n in NUMELS:
        n -= n % ws                       # reduce_scatter needs ws | numel
        inp = torch.ones(n, dtype=torch.float32, device=dev)
        out = torch.empty(n // ws, dtype=torch.float32, device=dev)
        dist.barrier()
        t0 = time.time()
        try:
            _rs(out, inp)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            elif dev.type == "xpu":
                torch.xpu.synchronize()
            dt = time.time() - t0
            ok = bool(out.eq(float(ws)).all().item())   # every rank sent 1s
            v = "OK" if ok else "WRONG_VALUES"
        except Exception as e:                       # noqa: BLE001
            dt = time.time() - t0
            v = f"EXC:{type(e).__name__}"
        if rank == 0:
            print(f"{n}\t{n*4/2**20:.3f}\t{v}\t{dt:.2f}", flush=True)
        del inp, out

    if rank == 0:
        print("SWEEP_COMPLETE", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
