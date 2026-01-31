#!/usr/bin/env python

"""
ezpz/launch.py

Launch a command on the current PBS or SLURM job.

By default, the command to be executed will be launched across _all_ nodes.
"""

import os
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import ezpz
from ezpz.cli.flags import build_launch_parser

logger = ezpz.get_logger(__name__)


def _split_launch_and_command(
    argv: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split ezpz launch args from the command to execute at the first ``--``."""
    if "--" in argv:
        idx = list(argv).index("--")
        return list(argv[:idx]), list(argv[idx + 1 :])
    return list(argv), []


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
                # "operator: aten::geometric"
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
        # and "python" not in str(cmd_to_launch_list[0]):
        found_python = False
        for part in cmd_to_launch_list:
            if "python" in str(part):
                found_python = True
        if not found_python:
            cmd_to_launch_list.insert(0, sys.executable)
        # cmd_to_launch_list = [sys.executable] + cmd_to_launch_list

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


def launch(
    launch_cmd: Optional[str] = None,
    cmd_to_launch: Optional[str | list[str]] = None,
    include_python: bool = False,
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[str | os.PathLike | Path] = None,
    filters: Optional[list[str]] = None,
    launcher_args: Optional[Sequence[str]] = None,
) -> int:
    """Launch a command on the current {PBS, SLURM} job."""
    start = time.perf_counter()
    print("\n") if ezpz.get_rank() == 0 else None
    logger.info(f"----[🍋 ezpz.launch][started][{ezpz.get_timestamp()}]----")
    from ezpz.configs import get_json_log_file

    json_log_file = get_json_log_file()
    if json_log_file is not None:
        logger.info(f"Logs available at: {json_log_file}")
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

    filters = [] if filters is None else filters
    if ezpz.get_machine().lower() in {"aurora", "sunspot"}:
        filters += get_aurora_filters()

    logger.info(f"Execution started @ {ezpz.get_timestamp()}...")
    cmd_start = time.perf_counter()
    retcode = run_command(command=cmd, filters=filters)
    cmd_finish = time.perf_counter()
    from ezpz.configs import get_json_log_file

    json_log_file = get_json_log_file()
    if json_log_file is not None:
        logger.info(f"Logs available at: {json_log_file}")
    logger.info(f"----[🍋 ezpz.launch][stop][{ezpz.get_timestamp()}]----")
    logger.info(f"Execution finished with {retcode}.")
    logger.info(f"Executing finished in {cmd_finish - cmd_start:.2f} seconds.")
    logger.info(
        f"Took {time.perf_counter() - t0:.2f} seconds to run. Exiting."
    )
    return retcode


def run(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for launching commands with scheduler fallback."""
    import ezpz.dist

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

    if scheduler in {"pbs", "slurm"}:
        jobid = get_active_jobid()
        if jobid is not None:
            launch(
                cmd_to_launch=command_parts,
                include_python=False,
                ngpus=(args.nproc if args.nproc > -1 else None),
                nhosts=(args.nhosts if args.nhosts > -1 else None),
                ngpu_per_host=(
                    args.nproc_per_node if args.nproc_per_node > -1 else None
                ),
                hostfile=args.hostfile,
                filters=args.filter,
                launcher_args=getattr(args, "launcher_args", []),
            )
            ezpz.dist.cleanup()
            return 0

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
    fallback_cmd.extend(getattr(args, "launcher_args", []))
    fallback_cmd.extend(command_parts)
    filters = getattr(args, "filters", [])
    if ezpz.get_machine().lower() in {"aurora", "sunspot"}:
        filters += get_aurora_filters()

    print("\n") if ezpz.get_rank() == 0 else None
    logger.info(f"----[🍋 ezpz.launch][started][{ezpz.get_timestamp()}]----")
    from ezpz.configs import get_json_log_file

    json_log_file = get_json_log_file()
    if json_log_file is not None:
        logger.info(f"Logs available at: {json_log_file}")
    logger.info(
        "No active scheduler detected; falling back to local mpirun: %s",
        " ".join(shlex.quote(part) for part in fallback_cmd),
    )
    logger.info(f"Execution started @ {ezpz.get_timestamp()}...")
    cmd_start = time.perf_counter()
    if filters:
        retcode = run_command(command=fallback_cmd, filters=filters)
    else:
        proc = subprocess.run(fallback_cmd, check=False)
        retcode = proc.returncode
    cmd_finish = time.perf_counter()
    from ezpz.configs import get_json_log_file

    json_log_file = get_json_log_file()
    if json_log_file is not None:
        logger.info(f"Logs available at: {json_log_file}")
    logger.info(f"----[🍋 ezpz.launch][stop][{ezpz.get_timestamp()}]----")
    logger.info(f"Execution finished with {retcode}.")
    logger.info(f"Executing finished in {cmd_finish - cmd_start:.2f} seconds.")
    ezpz.dist.cleanup()
    return retcode


def main() -> int:
    """Backward-compatible console script entry point."""
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
