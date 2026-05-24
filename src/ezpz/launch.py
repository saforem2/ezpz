#!/usr/bin/env python

"""
ezpz/launch.py

Launch a command on the current PBS or SLURM job.

By default, the command to be executed will be launched across _all_ nodes.
"""

import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import ezpz
from ezpz.cli.flags import build_launch_parser
from ezpz.configs import get_json_log_file

logger = ezpz.get_logger(__name__)

# Exit code returned when the idle-stdout watchdog kills a process.
# Matches GNU coreutils `timeout(1)` convention so users can `$? == 124`
# to detect "killed because it went silent".
_WATCHDOG_EXIT_CODE = 124

# Grace period between SIGTERM and SIGKILL when the watchdog fires.
# Gives well-behaved processes a chance to flush logs and shut down
# cleanly; SIGKILL is the hammer for anything stuck.
_WATCHDOG_KILL_GRACE_S = 10.0

# Backoff cap between retry attempts (5s, 10s, 20s, 40s, then capped).
_RETRY_BACKOFF_CAP_S = 60.0


def _run_with_watchdog(
    cmd: Sequence[str], idle_timeout_s: Optional[int]
) -> int:
    """Run *cmd* with an optional idle-output watchdog.

    When ``idle_timeout_s is None`` (or ``<= 0``) this is a thin
    pass-through to ``subprocess.run`` — no threading, no overhead.

    Otherwise: spawn the process, stream its combined stdout+stderr
    to this process's stdout in real time, and SIGTERM it (then
    SIGKILL after ``_WATCHDOG_KILL_GRACE_S``) if no output appears
    for ``idle_timeout_s`` consecutive seconds. Returns the process's
    exit code, or ``_WATCHDOG_EXIT_CODE`` (124) if the watchdog fired.

    Idle = silence on *both* stdout and stderr (stderr is merged into
    stdout). The process can run indefinitely as long as it keeps
    emitting at least one line per ``idle_timeout_s`` on either
    stream. Designed for catching collective hangs (xccl, NCCL, etc.)
    where the process is alive but every rank is blocked in the same
    collective and nothing reaches either stream.

    **Buffering note**: forces ``PYTHONUNBUFFERED=1`` in the child
    environment. CPython block-buffers stdout when it isn't attached
    to a TTY (which it isn't here — we attached a pipe), and a healthy
    Python training script can easily produce zero bytes for minutes
    while accumulating a fat buffer. Without this nudge the watchdog
    would routinely kill perfectly-healthy jobs. The variable is
    benign for non-Python children: they ignore it.
    """
    if idle_timeout_s is None or idle_timeout_s <= 0:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode

    # Force unbuffered stdout in the child. See the buffering note in
    # the docstring — without this, Python children that go through
    # `print()` get block-buffered (4-8KB) and emit nothing for long
    # stretches even while actively running, fooling the watchdog.
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    # Merge stderr into stdout so a single reader thread covers both.
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered on our side; child uses PYTHONUNBUFFERED
        env=child_env,
    )

    last_activity = time.monotonic()
    activity_lock = threading.Lock()
    reader_done = threading.Event()

    def _drain_output() -> None:
        """Pump child output to ours, updating the activity timestamp."""
        nonlocal last_activity
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                with activity_lock:
                    last_activity = time.monotonic()
        finally:
            reader_done.set()

    threading.Thread(target=_drain_output, daemon=True).start()

    # Poll the process and the activity deadline. `try/finally`
    # ensures the reader thread gets one final chance to flush
    # whichever way we exit (clean exit, watchdog kill, exception).
    try:
        while True:
            rc = process.poll()
            if rc is not None:
                return rc

            with activity_lock:
                idle_for = time.monotonic() - last_activity

            if idle_for >= idle_timeout_s:
                logger.error(
                    "Watchdog: no output for %.1fs (timeout=%ds). "
                    "Sending SIGTERM to PID %d.",
                    idle_for,
                    idle_timeout_s,
                    process.pid,
                )
                process.terminate()
                try:
                    process.wait(timeout=_WATCHDOG_KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    logger.error(
                        "Watchdog: PID %d still alive %ds after SIGTERM. "
                        "Sending SIGKILL.",
                        process.pid,
                        int(_WATCHDOG_KILL_GRACE_S),
                    )
                    process.kill()
                    process.wait()
                return _WATCHDOG_EXIT_CODE

            # Sleep until the soonest of (process exit, idle deadline).
            # Cap sleep so we still notice process exit promptly.
            sleep_for = min(1.0, max(0.1, idle_timeout_s - idle_for))
            time.sleep(sleep_for)
    finally:
        # Always give the reader thread a chance to flush remaining
        # buffered output — covers normal exit, watchdog kill, and any
        # unexpected exception out of the poll loop.
        reader_done.wait(timeout=2.0)


def _run_with_retries(
    cmd: Sequence[str],
    idle_timeout_s: Optional[int],
    retries: int,
) -> int:
    """Wrap :func:`_run_with_watchdog` in a non-zero-exit retry loop.

    Re-executes *cmd* up to ``retries`` additional times when the
    previous attempt returns a non-zero exit code (including the
    watchdog's 124). Applies exponential backoff between attempts:
    5s, 10s, 20s, 40s, then capped at ``_RETRY_BACKOFF_CAP_S``.
    """
    max_attempts = max(1, retries + 1)
    last_rc = 0
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            backoff = min(_RETRY_BACKOFF_CAP_S, 5.0 * (2 ** (attempt - 2)))
            logger.warning(
                "Retry %d/%d (prior exit=%d). Waiting %.0fs before relaunch...",
                attempt - 1,
                max_attempts - 1,
                last_rc,
                backoff,
            )
            time.sleep(backoff)
            logger.info("Retry attempt %d/%d", attempt, max_attempts)
        last_rc = _run_with_watchdog(cmd, idle_timeout_s)
        if last_rc == 0:
            if attempt > 1:
                logger.info(
                    "Retry attempt %d succeeded after %d prior failure(s).",
                    attempt,
                    attempt - 1,
                )
            return 0
    return last_rc


def _split_launch_and_command(
    argv: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split ezpz launch args from the command to execute at the first ``--``."""
    if "--" in argv:
        idx = list(argv).index("--")
        return list(argv[:idx]), list(argv[idx + 1 :])
    return list(argv), []


def _resolve_launch_python() -> str:
    """Pick the python interpreter to prefix into the launch command.

    Resolution order:

    1. ``$VIRTUAL_ENV/bin/python3`` if it exists
    2. ``$VIRTUAL_ENV/bin/python`` if it exists (some envs lack the
       versioned symlink)
    3. ``shutil.which("python3")`` — picks up conda envs and pyenv
       shims via PATH
    4. ``sys.executable`` — last resort

    ``sys.executable`` is *intentionally* the last fallback: it's
    frozen at interpreter startup and on HPC clusters often points
    to a Lustre-resident venv that the user has since copied to
    ``/tmp`` (via ``ezpz yeet-env``).  Returning that stale path
    re-imports modules from Lustre and defeats the whole point of
    the local copy.  Both ``$VIRTUAL_ENV/bin/python*`` and a fresh
    ``shutil.which`` reflect the current ``activate``-d environment.
    """
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        for name in ("python3", "python"):
            candidate = os.path.join(venv, "bin", name)
            if os.path.isfile(candidate):
                return candidate
    return shutil.which("python3") or sys.executable


def command_exists(cmd: str) -> bool:
    """Return True when the command is discoverable on PATH."""
    from ezpz.configs import command_exists as _command_exists

    return _command_exists(cmd)


def get_scheduler(_scheduler: Optional[str] = None) -> str:
    """Delegate scheduler detection to the configs module."""
    from ezpz.configs import get_scheduler as _get_scheduler

    return _get_scheduler(_scheduler=_scheduler)


def run_bash_command(command: str) -> subprocess.CompletedProcess[str]:
    """Execute a bash command and capture its output."""

    return subprocess.run(
        command,
        shell=True,
        check=False,
        text=True,
        capture_output=True,
    )


def _log_json_log_file(active_logger) -> None:
    json_log_file = get_json_log_file()
    if json_log_file is not None:
        active_logger.info("Logs available at: %s", json_log_file)


EZPZ_LOG_LEVEL: str = os.environ.get("EZPZ_LOG_LEVEL", "INFO").upper()


def parse_args(argv: Optional[Sequence[str]] = None):
    """Parse command line arguments."""
    argv = [] if argv is None else list(argv)
    launch_argv, command_from_sep = _split_launch_and_command(argv)
    if any(flag in launch_argv for flag in ("-h", "--help")):
        # Show help with the positional command documented.
        parser = build_launch_parser(include_command=True)
        parser.parse_args(launch_argv)
    parser = build_launch_parser(include_command=False)
    args, unknown = parser.parse_known_args(launch_argv)
    args.command = command_from_sep if command_from_sep else unknown
    # Unknown flags that precede the ``--`` separator are forwarded to the
    # underlying launcher (e.g., mpirun -x FOO=bar -- python ...).
    args.launcher_args = unknown if command_from_sep else []

    # Cross-flag validation. argparse mutex groups can't express
    # "--auto-retry vs --retries with a non-zero value" (since
    # --retries defaults to 0), so do it here. Same for --auto-retry
    # requiring an explicit --nproc — the value depends on whether
    # the user passed -n at all, which argparse can't observe from
    # inside the parser.
    if getattr(args, "auto_retry", False):
        if getattr(args, "retries", 0):
            raise SystemExit(
                "--auto-retry is mutually exclusive with --retries. "
                "--retries is bounded per-process retry; --auto-retry "
                "is unbounded node-level failover. Pick one."
            )
        if getattr(args, "nproc", -1) <= 0:
            raise SystemExit(
                "--auto-retry requires --nproc (-n/--np) to be set "
                "explicitly. The auto-retry loop needs the training "
                "rank count to split the PBS allocation into active "
                "+ spare hosts."
            )
    return args


def _normalize_command(command: Sequence[str] | str) -> list[str]:
    """Return a list suitable for ``subprocess`` from *command*."""

    if isinstance(command, str):
        return shlex.split(command)
    return list(command)


def _mpirun_supports_flag(flag: str) -> bool:
    try:
        proc = subprocess.run(
            ["mpirun", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    output = f"{proc.stdout}\n{proc.stderr}"
    return flag in output


def _get_mpirun_env_flags() -> list[str]:
    if _mpirun_supports_flag("--envall"):
        return ["--envall"]
    if _mpirun_supports_flag("-envall"):
        return ["-envall"]
    env_keys = []
    if os.environ.get("TORCH_DEVICE"):
        env_keys.append("TORCH_DEVICE")
    if _mpirun_supports_flag("-x") and env_keys:
        flags: list[str] = []
        for key in env_keys:
            flags.extend(["-x", key])
        return flags
    return []


def _normalize_cpu_bind_value(cpu_bind: Optional[str]) -> Optional[str]:
    """Normalize user-provided CPU bind values to a launcher value."""
    if cpu_bind is None:
        return None
    normalized = cpu_bind.strip().removeprefix("--cpu-bind=").strip()
    return normalized or None


def _cpu_bind_launcher_args(cpu_bind: Optional[str]) -> list[str]:
    """Render launcher args for CPU bind, if provided."""
    if cpu_bind is None:
        return []
    return [f"--cpu-bind={cpu_bind}"]


def run_command(
    command: Sequence[str] | str, filters: Optional[Sequence[str]] = None
) -> int:
    """Run a command and print its output line by line.

    Args:

    - command (str or list): The command to run. If a string, it will be split
      into a list
    - filters (list, optional): A list of strings to filter the output
      lines.
    """
    # XXX: Replace `subprocess.Popen`
    # with `subprocess.run` for better error handling ??
    # <https://docs.python.org/3.10/library/subprocess.html#subprocess.run>
    cmd_list = _normalize_command(command)

    if filters is not None and len(filters) > 0:
        logger.info(f"Caught {len(filters)} filters")
    logger.info(
        " ".join(
            [
                "Running command:\n",
                shlex.join(cmd_list),
            ]
        )
    )
    os.environ["EZPZ_RUN_COMMAND"] = str(cmd_list)
    with subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        close_fds=True,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            if (
                filters is None
                or len(filters) == 0
                or not any(f in line for f in filters)
            ):
                print(line.rstrip())
    return process.returncode or 0


def get_command_to_launch_from_argv() -> Optional[str | list[str]]:
    """Return the command specified on ``sys.argv`` or ``None`` if absent."""
    assert len(sys.argv) > 1, "No command to run."
    # cmd_to_launch = shlex.join(sys.argv[1:])
    cmd_to_launch = " ".join(sys.argv[1:])

    return cmd_to_launch


def configure_warnings():
    """Silence noisy deprecation warnings for child processes."""
    import os
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:__main__"


def get_aurora_filters(additional_filters: Optional[list] = None) -> list:
    """Return log filtering patterns tailored for Aurora clusters."""
    mn = ezpz.get_machine()
    filters = [*additional_filters] if additional_filters else []
    if mn.lower() == "aurora":
        if EZPZ_LOG_LEVEL == "DEBUG":
            filters = []
        else:
            filters += [
                "cuda",
                "CUDA",
                "cuDNN",
                "cuBLAS",
                "[W501",
                "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'",
                "  Overriding a previously registered kernel",
                "operator: aten::_cummax_helper",
                "    registered at build",
                "dispatch key: XPU",
                "previous kernel: registered at",
                "pkg_resources is deprecated as an API",
                "import pkg_resources",
                "UserWarning: pkg_resources",
                "new kernel: registered at",
                "/build/pytorch/build/aten/src/ATen/RegisterSchema.cpp",
                "Setting ds_accelerator to xpu",
                "Trying to register 2 metrics with the same name",
                "TF-TRT Warning",
                "Warning only once",
                "measureDifference between two events",
                "AttributeError",
                "Initialized with serialization",
                "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'",
                "In file included from /var/run/palsd/",
                "/opt/aurora/26.26.0/oneapi/compiler/latest/include/sycl/sycl.hpp:41:1",
                '41 | __SYCL_WARNING("You are including <sycl/sycl.hpp> without -fsycl flag',
                "| ^",
                "/opt/aurora/26.26.0/oneapi/compiler/latest/include/sycl/sycl.hpp:34:29: note:",
                "34 | #define __SYCL_WARNING(msg) _Pragma(__SYCL_TOSTRING(GCC warning msg))",
                # |                             ^
                "<scratch space>:58:6: note: expanded from here",
                '58 |  GCC warning "You are including <sycl/sycl.hpp> without -fsycl flag, which is errorenous for device code compilation.',
                "This warning can be disabled by setting SYCL_DISABLE_FSYCL_SYCLHPP_WARNING macro.",
                "|      ^",
                '# "operator: aten::geometric"',
                "1 warning generated.",
                "|                             ^",
                '41 | __SYCL_WARNING("You are including <sycl/sycl.hpp> without -fsycl flcan be disabled by setting SYCL_DISABLE_FSYCL_SYCLHPP_WARNING macro."',
                "ag, \\",
                "34 | #define __SYCL_WARNING(msg) _Pragma(__SYCL_TOmacro.",
                "STRING(GCC warning msg))",
                '41 | __SYCL_WARNING("are including <sycl/sycl.hpp> without -fsycl flag, \\',
                "by setting SYCL_DISABLE_FSYCL_SYCLHPP_WARNING macro.",
            ]
        logger.info(
            " ".join(
                [
                    "Filtering for Aurora-specific messages.",
                    "To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG",
                ]
            )
        )
    logger.debug(f"Filters: {filters}")
    return filters


def kill_existing_processes(
    filters: Optional[list] = None,
    additional_filters: Optional[list] = None,
) -> int:
    """Kill existing processes that match the filters."""
    # TODO: Run this as preamble to launching
    filters = [] if filters is None else filters
    if ezpz.get_machine().lower() == "aurora":
        filters += get_aurora_filters(additional_filters=additional_filters)

    if len(filters) == 0:
        logger.info("No filters provided; skipping process cleanup.")
        return 0

    logger.info(f"Killing existing processes with filters: {filters}")
    filter_pattern = " ".join(filters)
    cmd = ["pkill", "-f", filter_pattern]
    return run_command(cmd, filters=filters)


def get_active_jobid() -> str | None:
    """Return the job identifier for the currently running PBS/SLURM job."""
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler().lower()
    if scheduler == "pbs":
        import ezpz.pbs

        return ezpz.pbs.get_pbs_jobid_of_active_job()
    elif scheduler == "slurm":
        import ezpz.slurm

        return ezpz.slurm.get_slurm_jobid_of_active_job()
    else:
        return None


def get_nodelist_of_active_job() -> list[str] | None:
    """Get nodelist of active job."""
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler().lower()
    if scheduler == "pbs":
        import ezpz.pbs

        jobid = ezpz.pbs.get_pbs_jobid_of_active_job()
        if jobid is not None:
            return ezpz.pbs.get_pbs_nodelist_from_jobid(jobid)
    elif scheduler == "slurm":
        import ezpz.slurm

        jobid = ezpz.slurm.get_slurm_jobid_of_active_job()
        if jobid is not None:
            return ezpz.slurm.get_nodelist_from_slurm_jobid(jobid)
    return None


def get_hostfile_of_active_job():
    """Get hostfile of active job."""
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler().lower()
    if scheduler == "pbs":
        import ezpz.pbs

        return ezpz.pbs.get_pbs_nodefile_of_active_job()
    elif scheduler == "slurm":
        import ezpz.slurm

        # jobid = ezpz.slurm.get_slurm_jobid_of_active_job()
        # if jobid is not None:
        #     return ezpz.slurm.get_slurm_nodefile_from_jobid(jobid)
        return ezpz.slurm.get_slurm_nodefile_of_active_job()
    return None


def build_executable(
    launch_cmd: Optional[str] = None,
    cmd_to_launch: Optional[str | list[str]] = None,
    include_python: bool = False,
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[str | os.PathLike | Path] = None,
    cpu_bind: Optional[str] = None,
    extra_launch_args: Optional[Sequence[str]] = None,
) -> list:
    """Build the full executable command to launch.

    Args:
        launch_cmd (str, optional): The command to launch the job. If None,
            will be built using `build_launch_cmd()`.
        cmd_to_launch (str or list, optional): The command to run on the job.
            If None, will be taken from `sys.argv`.
        include_python (bool, optional): Whether to include the python
            executable in the command. Defaults to False.
        extra_launch_args (Sequence[str], optional): Additional arguments to
            append to the scheduler/launcher invocation (e.g., mpirun flags).

    Returns:
        list[str]: The full command to launch the job.
    """
    extra_launch_args = list(extra_launch_args) if extra_launch_args else []
    from ezpz.pbs import build_launch_cmd

    launch_cmd = (
        build_launch_cmd(
            ngpus=ngpus,
            nhosts=nhosts,
            ngpu_per_host=ngpu_per_host,
            hostfile=hostfile,
            cpu_bind=cpu_bind,
        )
        if launch_cmd is None
        else launch_cmd
    )
    cmd_to_launch = (
        get_command_to_launch_from_argv()
        if cmd_to_launch is None
        else cmd_to_launch
    )
    cmd_to_launch_list: list[str] = (
        shlex.split(cmd_to_launch)
        if isinstance(cmd_to_launch, str)
        else (cmd_to_launch if cmd_to_launch is not None else [])
    )
    if include_python:
        found_python = any("python" in str(p) for p in cmd_to_launch_list)
        if not found_python:
            cmd_to_launch_list.insert(0, _resolve_launch_python())

    cmd_to_launch_str = shlex.join(cmd_to_launch_list)
    logger.info("Building command to execute by piecing together:")
    logger.info(f"(1.) launch_cmd: {launch_cmd}")
    logger.info(f"(2.) cmd_to_launch: {cmd_to_launch_str}")
    executable = [
        *shlex.split(launch_cmd),
        *extra_launch_args,
        *cmd_to_launch_list,
    ]
    # executable = [
    #     shlex.join(launch_cmd.split(' ')), *cmd_to_launch_list
    # ]
    # return shlex.split(shlex.join(executable))
    return executable


def _resolve_auto_retry_allocation(
    full_nodelist: Sequence[str],
    nproc_active_hosts: int,
    spare_nodes: "int | str | None",
    log_dir: Path,
):
    """Build a :class:`NodeAllocation` for ``--auto-retry``.

    Validates the spare-node policy against the actual PBS allocation
    and persists the active hostfile so the launcher (re-spawned each
    attempt with the same path) sees the current active set.

    ``spare_nodes`` semantics match the CLI flag:
      * ``"auto"``  → derive ``total - nproc_active_hosts``
      * ``int``     → explicit count (must fit in spare pool)
      * ``None``    → also ``auto`` (default when --auto-retry is set)
    """
    from ezpz.launch_autoretry import NodeAllocation, derive_spare_count

    total = len(full_nodelist)
    if nproc_active_hosts > total:
        raise SystemExit(
            f"--auto-retry: need {nproc_active_hosts} active hosts but "
            f"PBS allocation only has {total}"
        )

    spare_default = derive_spare_count(total, nproc_active_hosts)
    if spare_nodes is None or spare_nodes == "auto":
        resolved_spare = spare_default
    else:
        resolved_spare = int(spare_nodes)
        if resolved_spare > spare_default:
            raise SystemExit(
                f"--auto-retry: requested {resolved_spare} spare nodes "
                f"but allocation only has {spare_default} unused "
                f"(total={total}, active={nproc_active_hosts})"
            )

    # We slice the nodelist active-first, spare-second — only the
    # first nproc+resolved_spare hosts get used. Beyond that, the
    # PBS-allocated hosts just sit idle (consistent with how
    # failover.sh's `failover_init` truncates).
    slice_len = nproc_active_hosts + resolved_spare
    relevant_nodes = list(full_nodelist[:slice_len])

    log_dir.mkdir(parents=True, exist_ok=True)
    hostfile_path = log_dir / "active.hostfile"
    bad_nodes_path = log_dir / "bad_nodes.txt"

    allocation = NodeAllocation.from_full_nodelist(
        relevant_nodes,
        nproc_active_hosts,
        hostfile_path,
        bad_nodes_path,
    )
    logger.info(
        "[auto-retry] %d total / %d active / %d spare. Active hostfile: %s",
        total,
        len(allocation.active),
        len(allocation.spare),
        hostfile_path,
    )
    return allocation, hostfile_path


def _auto_retry_log_dir(jobid: str) -> Path:
    """Per-job log directory for attempt-N.log + bad_nodes.txt.

    Matches the bash lib's ``$FAILOVER_LOG_DIR`` shape
    (``$(pwd)/logs/failover-<short_jobid>``) so users grepping
    postmortems don't need to learn a second convention.
    """
    short = jobid.split(".", 1)[0]
    return Path.cwd() / "logs" / f"failover-{short}"


def launch(
    launch_cmd: Optional[str] = None,
    cmd_to_launch: Optional[str | list[str]] = None,
    include_python: bool = False,
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[str | os.PathLike | Path] = None,
    cpu_bind: Optional[str] = None,
    filters: Optional[list[str]] = None,
    launcher_args: Optional[Sequence[str]] = None,
    idle_timeout_s: Optional[int] = None,
    retries: int = 0,
    auto_retry: bool = False,
    spare_nodes: "int | str | None" = None,
    max_failover_retries: Optional[int] = None,
) -> int:
    """Launch a command on the current {PBS, SLURM} job."""
    start = time.perf_counter()
    print("\n") if ezpz.get_rank() == 0 else None
    logger.info(f"----[🍋 ezpz.launch][started][{ezpz.get_timestamp()}]----")
    _log_json_log_file(logger)
    jobid = get_active_jobid()
    assert jobid is not None, "No active job found."
    nodelist = get_nodelist_of_active_job()
    active_hostfile = get_hostfile_of_active_job()
    selected_hostfile: Optional[Path]
    if hostfile is not None:
        selected_hostfile = Path(hostfile).expanduser()
    else:
        selected_hostfile = (
            Path(active_hostfile).expanduser()
            if active_hostfile is not None
            else None
        )
    if selected_hostfile is not None and not selected_hostfile.exists():
        logger.warning(
            "Hostfile %s does not exist; continuing without explicit hostfile.",
            selected_hostfile,
        )
        selected_hostfile = None

    # --auto-retry path: split the full nodelist into active + spare
    # BEFORE building the launcher command, so the launcher's
    # topology inference sees the (smaller) active hostfile and we
    # don't trip _infer_topology's "ngpus > N_active*ppn" check on
    # the unused spare hosts.
    autoretry_allocation = None
    autoretry_log_dir: Optional[Path] = None
    if auto_retry:
        if ngpus is None:
            raise SystemExit(
                "--auto-retry requires --nproc (-n/--np) to be set "
                "explicitly. We need to know the training rank count "
                "to split the PBS allocation into active + spare."
            )
        if nodelist is None or not nodelist:
            raise SystemExit(
                "--auto-retry: failed to read the PBS nodelist (no "
                "active job?). Cannot split into active + spare."
            )
        gpus_per_node = ngpu_per_host or ezpz.get_gpus_per_node() or 1
        # Round-up division: 12 ranks on 4-gpu nodes → 3 nodes.
        nhosts_active = (ngpus + gpus_per_node - 1) // gpus_per_node
        autoretry_log_dir = _auto_retry_log_dir(jobid)
        autoretry_allocation, autoretry_hostfile = (
            _resolve_auto_retry_allocation(
                nodelist,
                nhosts_active,
                spare_nodes,
                autoretry_log_dir,
            )
        )
        # Override: subsequent build_executable + topology inference
        # see the smaller active hostfile, not the full PBS aux file.
        selected_hostfile = autoretry_hostfile
        nhosts = nhosts_active

    logger.info(f"Job ID: {jobid}")
    logger.info(f"nodelist: {nodelist}")
    logger.info(f"hostfile: {selected_hostfile}")
    cmd_list = build_executable(
        launch_cmd=launch_cmd,
        cmd_to_launch=cmd_to_launch,
        ngpus=ngpus,
        ngpu_per_host=ngpu_per_host,
        nhosts=nhosts,
        include_python=include_python,
        hostfile=selected_hostfile,
        cpu_bind=cpu_bind,
        extra_launch_args=launcher_args,
    )
    # cmd_list = shlex.split(cmd)
    cmd_str = shlex.join([f"{i}" for i in cmd_list])
    cmd = shlex.split(cmd_str)

    logger.info(
        f"Took: {time.perf_counter() - start:.2f} seconds to build command."
    )
    logger.info("Executing:\n" + "\n  ".join([f"{i}" for i in cmd_list]))
    t0 = time.perf_counter()

    os.environ["EZPZ_RUN_COMMAND"] = str(cmd)
    logger.info(f"Execution started @ {ezpz.get_timestamp()}...")
    cmd_start = time.perf_counter()
    if autoretry_allocation is not None:
        from ezpz.launch_autoretry import (
            AutoRetryConfig,
            DEFAULT_AUTO_RETRY_IDLE_TIMEOUT_S,
            run_with_auto_retry,
        )

        assert autoretry_log_dir is not None  # set above when auto_retry
        # Default --auto-retry's idle watchdog to 30min if the caller
        # didn't pass --timeout — matches FAILOVER_IDLE_TIMEOUT in
        # failover.sh and prevents 5h xccl hangs from burning the
        # full PBS walltime.
        effective_timeout = (
            idle_timeout_s
            if idle_timeout_s is not None
            else DEFAULT_AUTO_RETRY_IDLE_TIMEOUT_S
        )
        ar_config = AutoRetryConfig(
            cmd=list(cmd),
            hostfile=Path(selected_hostfile)
            if selected_hostfile is not None
            else Path(),
            log_dir=autoretry_log_dir,
            idle_timeout_s=effective_timeout,
            max_failover_retries=max_failover_retries,
        )
        retcode = run_with_auto_retry(ar_config, autoretry_allocation)
    else:
        retcode = _run_with_retries(
            cmd, idle_timeout_s=idle_timeout_s, retries=retries
        )
    cmd_finish = time.perf_counter()
    _log_json_log_file(logger)
    logger.info(f"----[🍋 ezpz.launch][stop][{ezpz.get_timestamp()}]----")
    logger.info(f"Execution finished with {retcode}.")
    logger.info(f"Executing finished in {cmd_finish - cmd_start:.2f} seconds.")
    logger.info(
        f"Took {time.perf_counter() - t0:.2f} seconds to run. Exiting."
    )
    return retcode


def run(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for launching commands with scheduler fallback."""
    import ezpz.distributed

    configure_warnings()
    argv = [] if argv is None else list(argv)
    args = parse_args(argv)
    command_parts = [part for part in args.command if part]
    if not command_parts:
        if getattr(args, "print_source", False):
            from importlib import import_module

            launch_cli_mod = import_module("ezpz.cli.launch_cmd")
            source_path = Path(
                getattr(launch_cli_mod, "__file__", "")
            ).resolve()
            print(source_path)
            return 0
        raise SystemExit("No command provided to ezpz launch")

    scheduler = get_scheduler().lower()
    cli_cpu_bind = _normalize_cpu_bind_value(getattr(args, "cpu_bind", None))
    env_cpu_bind = _normalize_cpu_bind_value(os.environ.get("CPU_BIND"))
    selected_cpu_bind = cli_cpu_bind or env_cpu_bind
    if cli_cpu_bind is not None and env_cpu_bind is not None:
        logger.warning(
            "Both --cpu-bind and CPU_BIND are specified. "
            "Precedence order is: --cpu-bind > CPU_BIND. "
            "Using --cpu-bind=%s.",
            cli_cpu_bind,
        )

    if scheduler in {"pbs", "slurm"}:
        jobid = get_active_jobid()
        if jobid is not None:
            launcher_args = list(getattr(args, "launcher_args", []))
            if scheduler != "pbs":
                launcher_args.extend(
                    _cpu_bind_launcher_args(selected_cpu_bind)
                )
            rc = launch(
                cmd_to_launch=command_parts,
                include_python=False,
                ngpus=(args.nproc if args.nproc > -1 else None),
                nhosts=(args.nhosts if args.nhosts > -1 else None),
                ngpu_per_host=(
                    args.nproc_per_node if args.nproc_per_node > -1 else None
                ),
                hostfile=args.hostfile,
                cpu_bind=cli_cpu_bind if scheduler == "pbs" else None,
                filters=args.filter,
                launcher_args=launcher_args,
                idle_timeout_s=getattr(args, "idle_timeout_s", None),
                retries=getattr(args, "retries", 0),
                auto_retry=getattr(args, "auto_retry", False),
                spare_nodes=getattr(args, "spare_nodes", None),
                max_failover_retries=getattr(
                    args, "max_failover_retries", None
                ),
            )
            ezpz.distributed.cleanup()
            return rc

    requested_nproc = args.nproc if args.nproc > -1 else None
    requested_ppn = args.nproc_per_node if args.nproc_per_node > -1 else None
    requested_nhosts = args.nhosts if args.nhosts > -1 else None
    if (
        requested_nproc is None
        and requested_ppn is not None
        and requested_nhosts is not None
    ):
        requested_nproc = requested_ppn * requested_nhosts
    if requested_nproc is None:
        requested_nproc = int(os.environ.get("WORLD_SIZE", "2"))
    env_flags = _get_mpirun_env_flags()
    fallback_cmd = ["mpirun", *env_flags, "-np", str(requested_nproc)]
    if args.hostfile:
        fallback_cmd.extend(["--hostfile", args.hostfile])
    if requested_ppn is not None and requested_nhosts is not None:
        fallback_cmd.extend(["--map-by", f"ppr:{requested_ppn}:node"])
    fallback_cmd.extend(_cpu_bind_launcher_args(selected_cpu_bind))
    fallback_cmd.extend(getattr(args, "launcher_args", []))
    fallback_cmd.extend(command_parts)

    print("\n") if ezpz.get_rank() == 0 else None
    logger.info(f"----[🍋 ezpz.launch][started][{ezpz.get_timestamp()}]----")
    _log_json_log_file(logger)
    logger.info(
        "No active scheduler detected; falling back to local mpirun: %s",
        " ".join(shlex.quote(part) for part in fallback_cmd),
    )
    os.environ["EZPZ_RUN_COMMAND"] = " ".join(fallback_cmd)
    logger.info(f"Execution started @ {ezpz.get_timestamp()}...")
    cmd_start = time.perf_counter()
    retcode = _run_with_retries(
        fallback_cmd,
        idle_timeout_s=getattr(args, "idle_timeout_s", None),
        retries=getattr(args, "retries", 0),
    )
    cmd_finish = time.perf_counter()
    _log_json_log_file(logger)
    logger.info(f"----[🍋 ezpz.launch][stop][{ezpz.get_timestamp()}]----")
    logger.info(f"Execution finished with {retcode}.")
    logger.info(f"Executing finished in {cmd_finish - cmd_start:.2f} seconds.")
    ezpz.distributed.cleanup()
    return retcode


def main() -> int:
    """Backward-compatible console script entry point."""
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
