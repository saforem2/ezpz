"""Is it the SCALE of FSDP2's communicator state, or the calling thread?

Two refutations already stand (2026-09-18, 2026-09-19): neither the 4 MiB
payload nor two-stream AG/RS overlap triggers #239 on its own. Both prior
probes shared one weakness -- they reuse TWO buffers on the MAIN thread,
while FSDP2 cycles 14 distinct per-unit buffers and issues its parameter
all-gathers from AUTOGRAD ENGINE threads.

Arm E varies buffer count, F varies the calling thread, G does both.
Whichever hangs names the ingredient; if none do, the trigger is not
reachable without real FSDP2 and the ticket keeps the LoRA reproducer.
"""
import os, sys, threading, time
import torch, torch.distributed as dist

NUMEL = 1055232      # the r17 payload
N_UNITS = 14         # what fsdp_tp.py wraps: embed + 12 blocks + [norm,out]
ITERS = 30


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
                env[want] = env[c]; break
    if not (env.get("RANK") and env.get("WORLD_SIZE")):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        env["RANK"] = str(comm.Get_rank()); env["WORLD_SIZE"] = str(comm.Get_size())
        if not env.get("LOCAL_RANK"):
            host = MPI.Get_processor_name(); hosts = comm.allgather(host)
            env["LOCAL_RANK"] = str(
                [i for i, h in enumerate(hosts) if h == host].index(comm.Get_rank()))
        if not env.get("MASTER_ADDR"):
            env["MASTER_ADDR"] = comm.bcast(MPI.Get_processor_name(), root=0)
    env.setdefault("LOCAL_RANK", "0"); env.setdefault("MASTER_PORT", "29591")


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
            else torch.xpu.synchronize if dev.type == "xpu" else (lambda: None))
    rs = getattr(dist, "reduce_scatter_single", dist.reduce_scatter_tensor)
    ag = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)
    n = NUMEL - NUMEL % ws

    if rank == 0:
        print(f"# backend={dist.get_backend()} ws={ws} dev={dev.type} "
              f"torch={torch.__version__}", flush=True)
        print(f"# numel={n} units={N_UNITS} iters={ITERS}", flush=True)
        print("# arm\tverdict\tsecs", flush=True)

    def report(name, t0, err=None):
        if rank == 0:
            print(f"{name}\t{'OK' if err is None else 'EXC:'+err}\t"
                  f"{time.time()-t0:.2f}", flush=True)

    # E: 14 DISTINCT buffer pairs, cycled -- FSDP2's per-unit allocation
    #    pattern, which stresses the plugin's registration cache rather
    #    than reusing one hot registered region.
    # Allocate/free one unit at a time rather than holding all 14 live: the
    # point is CYCLING distinct registrations past the plugin's cache, and
    # holding them all OOM-killed the host (4 ranks/node, each with a full
    # CUDA context) on the first attempt.
    dist.barrier(); t0 = time.time()
    try:
        for _ in range(ITERS):
            for _u in range(N_UNITS):
                u_in = torch.ones(n, dtype=torch.float32, device=dev)
                u_out = torch.empty(n // ws, dtype=torch.float32, device=dev)
                rs(u_out, u_in)
                del u_in, u_out
        sync(); report("E-14-cycled-buffers", t0)
    except Exception as e:
        report("E-14-cycled-buffers", t0, type(e).__name__)

    # F: issued from a NON-MAIN thread. FSDP2's post_backward runs on the
    #    autograd engine's threads, not the thread that called init.
    err_box = {"done": 0}
    def worker():
        try:
            a_in = torch.ones(n, dtype=torch.float32, device=dev)
            a_out = torch.empty(n // ws, dtype=torch.float32, device=dev)
            for _ in range(ITERS):
                rs(a_out, a_in)
                err_box["done"] += 1
            sync()
            # Guard against a vacuous pass: a sub-0.1s "OK" for 30 inter-node
            # collectives means the work did not happen. Verify the payload.
            if not bool(a_out.eq(float(ws)).all().item()):
                err_box["e"] = "WRONG_VALUES"
        except Exception as e:                      # noqa: BLE001
            err_box["e"] = type(e).__name__
    dist.barrier(); t0 = time.time()
    th = threading.Thread(target=worker); th.start(); th.join(timeout=300)
    if th.is_alive():
        report("F-nonmain-thread", t0, "HUNG_THREAD")
    else:
        report("F-nonmain-thread(%d/%d)" % (err_box["done"], ITERS),
               t0, err_box.get("e"))

    # G: both at once -- 14 buffers cycled from a non-main thread, with an
    #    all-gather overlapping from the main thread. Closest to FSDP2
    #    short of running FSDP2.
    # A second PG for the overlapping all-gather. Two threads issuing on ONE
    # PG is an ordering violation (NCCL requires identical issue order across
    # ranks), which aborted the step with no diagnostic on the first attempt.
    try:
        ag_pg = dist.new_group(ranks=list(range(ws)))
    except Exception:                               # noqa: BLE001
        ag_pg = None
    err2 = {"done": 0}
    g_in = torch.ones(n, dtype=torch.float32, device=dev)
    g_out = torch.empty(n // ws, dtype=torch.float32, device=dev)
    def worker2():
        try:
            for _ in range(ITERS):
                for _u in range(N_UNITS):
                    rs(g_out, g_in)
                err2["done"] += 1
            sync()
            if not bool(g_out.eq(float(ws)).all().item()):
                err2["e"] = "WRONG_VALUES"
        except Exception as e:                      # noqa: BLE001
            err2["e"] = type(e).__name__
    ag_in = torch.ones(n // ws, dtype=torch.float32, device=dev)
    ag_out = torch.empty(n, dtype=torch.float32, device=dev)
    dist.barrier(); t0 = time.time()
    th2 = threading.Thread(target=worker2); th2.start()
    try:
        for _ in range(ITERS):
            ag(ag_out, ag_in, group=ag_pg) if ag_pg is not None else None
    except Exception as e:                          # noqa: BLE001
        err2.setdefault("e", "main:" + type(e).__name__)
    th2.join(timeout=300)
    if th2.is_alive():
        report("G-cycled-nonmain+ag", t0, "HUNG_THREAD")
    else:
        report("G-cycled-nonmain+ag(%d/%d)" % (err2["done"], ITERS),
               t0, err2.get("e"))

    if rank == 0:
        print("UNITS_COMPLETE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
