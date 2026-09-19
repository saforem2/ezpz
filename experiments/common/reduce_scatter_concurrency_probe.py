"""Does CONCURRENCY trigger ezpz #239, when payload size alone does not?

Established 2026-09-18: a bare reduce_scatter_tensor at NumelIn=1055232 --
the exact buffer the LoRA job deadlocks on -- completes in <0.1s on
Perlmutter's aws-ofi-nccl/cxi path. So size is not sufficient.

What FSDP2 does that the bare sweep does not:
  A. issues all-gathers CONCURRENTLY with the reduce-scatter (prefetch of
     unit N+1's params overlaps unit N's gradient reduction);
  B. runs the two on DIFFERENT CUDA STREAMS sharing one communicator;
  C. interleaves them in a long repeating sequence, not a single shot.

Each arm below adds one of those to the same 4.025 MiB reduce-scatter, so
whichever arm hangs names the missing ingredient. Arms are independent and
each is time-boxed, so one job tests all of them even if an early arm dies.
"""
import os, sys, time
import torch, torch.distributed as dist

NUMEL = 1055232          # the r17 payload, verbatim
AG_NUMEL = 1055232       # comparable all-gather, overlapped with it
ITERS = 40               # FSDP2 does this every step, not once


def _derive_rank_env():
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
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        env["RANK"] = str(comm.Get_rank())
        env["WORLD_SIZE"] = str(comm.Get_size())
        if not env.get("LOCAL_RANK"):
            host = MPI.Get_processor_name()
            hosts = comm.allgather(host)
            env["LOCAL_RANK"] = str(
                [i for i, h in enumerate(hosts) if h == host].index(comm.Get_rank()))
        if not env.get("MASTER_ADDR"):
            env["MASTER_ADDR"] = comm.bcast(MPI.Get_processor_name(), root=0)
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("MASTER_PORT", "29581")


def main():
    _derive_rank_env()
    dist.init_process_group(backend=sys.argv[1] if len(sys.argv) > 1 else "nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{lr}"); torch.cuda.set_device(dev)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        dev = torch.device(f"xpu:{lr}"); torch.xpu.set_device(dev)
    else:
        dev = torch.device("cpu")
    sync = (torch.cuda.synchronize if dev.type == "cuda"
            else torch.xpu.synchronize if dev.type == "xpu"
            else (lambda: None))
    Stream = torch.cuda.Stream if dev.type == "cuda" else (
        torch.xpu.Stream if dev.type == "xpu" else None)
    stream_ctx = torch.cuda.stream if dev.type == "cuda" else (
        torch.xpu.stream if dev.type == "xpu" else None)
    rs = getattr(dist, "reduce_scatter_single", dist.reduce_scatter_tensor)
    ag = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)

    n = NUMEL - NUMEL % ws
    an = AG_NUMEL - AG_NUMEL % ws
    rs_in = torch.ones(n, dtype=torch.float32, device=dev)
    rs_out = torch.empty(n // ws, dtype=torch.float32, device=dev)
    ag_in = torch.ones(an // ws, dtype=torch.float32, device=dev)
    ag_out = torch.empty(an, dtype=torch.float32, device=dev)

    if rank == 0:
        print(f"# backend={dist.get_backend()} ws={ws} dev={dev.type} "
              f"torch={torch.__version__}", flush=True)
        print(f"# rs numel={n} ({n*4/2**20:.3f} MiB)  iters={ITERS}", flush=True)
        print("# arm\tverdict\tsecs", flush=True)

    def report(name, t0, err=None):
        if rank == 0:
            print(f"{name}\t{'OK' if err is None else 'EXC:'+err}\t"
                  f"{time.time()-t0:.2f}", flush=True)

    # A: sequential baseline -- reconfirms the known-good case in THIS job.
    dist.barrier(); t0 = time.time()
    try:
        for _ in range(ITERS):
            rs(rs_out, rs_in)
        sync(); report("A-sequential-rs", t0)
    except Exception as e:
        report("A-sequential-rs", t0, type(e).__name__)

    # B: all-gather issued back-to-back with the reduce-scatter, same stream.
    dist.barrier(); t0 = time.time()
    try:
        for _ in range(ITERS):
            ag(ag_out, ag_in)
            rs(rs_out, rs_in)
        sync(); report("B-interleaved-1stream", t0)
    except Exception as e:
        report("B-interleaved-1stream", t0, type(e).__name__)

    # C: the FSDP2 shape -- all-gather on a SEPARATE stream, overlapping the
    #    reduce-scatter, one communicator. This is the arm I expect to hang.
    if Stream is None:
        report("C-twostream-overlap", time.time(), "NoStreamAPI")
    else:
        side = Stream()
        dist.barrier(); t0 = time.time()
        try:
            for _ in range(ITERS):
                with stream_ctx(side):
                    ag(ag_out, ag_in)
                rs(rs_out, rs_in)
            sync(); report("C-twostream-overlap", t0)
        except Exception as e:
            report("C-twostream-overlap", t0, type(e).__name__)

    # D: C plus an explicit cross-stream dependency, which is what makes
    #    FSDP2's prefetch actually overlap rather than serialise.
    if Stream is not None:
        side = Stream()
        dist.barrier(); t0 = time.time()
        try:
            cur = (torch.cuda.current_stream() if dev.type == "cuda"
                   else torch.xpu.current_stream())
            for _ in range(ITERS):
                side.wait_stream(cur)
                with stream_ctx(side):
                    ag(ag_out, ag_in)
                rs(rs_out, rs_in)
                cur.wait_stream(side)
            sync(); report("D-twostream-waited", t0)
        except Exception as e:
            report("D-twostream-waited", t0, type(e).__name__)

    if rank == 0:
        print("CONCURRENT_COMPLETE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
