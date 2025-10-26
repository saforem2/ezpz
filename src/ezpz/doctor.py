"""Runtime diagnostics for cluster readiness and local environment health."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
import sys
from collections.abc import Iterable, Sequence
from typing import Callable, Final, Literal, Optional

import ezpz

Status = Literal["ok", "warning", "error"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Structured outcome for individual diagnostic checks."""

    name: str
    status: Status
    message: str
    remedy: Optional[str] = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "remedy": self.remedy,
        }


STATUS_PRIORITY: Final[dict[Status, int]] = {"ok": 0, "warning": 1, "error": 2}

logger = ezpz.get_logger(__name__)


def _command_exists(command: str, *, which: Callable[[str], Optional[str]] = shutil.which) -> bool:
    return which(command) is not None


def check_mpi(which: Callable[[str], Optional[str]] = shutil.which) -> CheckResult:
    """Verify mpi4py importability and presence of a launcher command."""
    try:
        from mpi4py import MPI  # noqa: F401
        mpi_available = True
    except Exception:  # pragma: no cover - exercised via negative paths
        mpi_available = False

    has_launcher = any(_command_exists(cmd, which=which) for cmd in ("mpiexec", "mpirun"))
    if mpi_available and has_launcher:
        return CheckResult(
            name="mpi",
            status="ok",
            message="mpi4py import succeeded and an MPI launcher was found.",
        )
    if mpi_available:
        return CheckResult(
            name="mpi",
            status="warning",
            message="mpi4py is importable, but no MPI launcher was detected on PATH.",
            remedy="Install mpiexec/mpirun or load the appropriate module before launching distributed jobs.",
        )
    if has_launcher:
        return CheckResult(
            name="mpi",
            status="warning",
            message="An MPI launcher is available, but mpi4py could not be imported.",
            remedy="Install mpi4py into the active environment so Python workers can join the MPI communicator.",
        )
    return CheckResult(
        name="mpi",
        status="error",
        message="Neither mpi4py nor a launcher (mpiexec/mpirun) is available.",
        remedy="Install mpi4py and ensure an MPI runtime is accessible on PATH.",
    )


def check_scheduler(
    *,
    get_scheduler: Callable[[], str] = ezpz.configs.get_scheduler,
    environ: Optional[dict[str, str]] = None,
) -> CheckResult:
    """Determine scheduler visibility from environment variables."""
    env = os.environ if environ is None else environ
    scheduler = get_scheduler()
    if scheduler in {"PBS", "SLURM"}:
        return CheckResult(
            name="scheduler",
            status="ok",
            message=f"Detected active scheduler: {scheduler}.",
        )
    suspect_vars = [key for key in ("PBS_JOBID", "SLURM_JOB_ID", "SLURM_JOBID") if env.get(key)]
    if suspect_vars:
        return CheckResult(
            name="scheduler",
            status="warning",
            message="Scheduler variables detected but mapping returned UNKNOWN.",
            remedy="Confirm ezpz.configs.get_scheduler recognises this host or provide a plug-in adapter.",
        )
    return CheckResult(
        name="scheduler",
        status="warning",
        message="No scheduler detected – assuming local launch mode.",
        remedy="Set scheduler environment variables or configure a custom adapter if running under a job queue.",
    )


def check_wandb(environ: Optional[dict[str, str]] = None) -> CheckResult:
    """Advise on Weights & Biases connectivity expectations."""
    env = os.environ if environ is None else environ
    try:
        import wandb  # type: ignore # pragma: no cover - optional dependency

        _ = wandb.__version__
        wandb_importable = True
    except Exception:
        wandb_importable = False

    api_key = env.get("WANDB_API_KEY")
    offline_mode = env.get("WANDB_MODE", "").lower() == "offline"

    if not wandb_importable:
        if api_key or not offline_mode:
            return CheckResult(
                name="wandb",
                status="warning",
                message="WANDB credentials present but the library could not be imported.",
                remedy="Install `ezpz[monitoring]` or set WANDB_MODE=offline to suppress remote logging.",
            )
        return CheckResult(
            name="wandb",
            status="ok",
            message="wandb not installed and no cloud logging requested.",
        )

    if offline_mode:
        return CheckResult(
            name="wandb",
            status="ok",
            message="wandb is available and offline logging is configured.",
        )
    if api_key:
        return CheckResult(
            name="wandb",
            status="ok",
            message="wandb is available and WANDB_API_KEY is set for cloud logging.",
        )
    return CheckResult(
        name="wandb",
        status="warning",
        message="wandb installed but WANDB_API_KEY is not configured.",
        remedy="Set WANDB_MODE=offline for air-gapped runs or export WANDB_API_KEY for remote tracking.",
    )


def check_torch_device() -> CheckResult:
    """Check torch availability and configured accelerator."""
    try:
        import torch
    except Exception:  # pragma: no cover - optional dependency
        return CheckResult(
            name="torch",
            status="error",
            message="PyTorch is not importable in the current environment.",
            remedy="Install torch (matching your accelerator) or activate the environment that provides it.",
        )

    env_device = os.environ.get("TORCH_DEVICE")
    if env_device:
        return CheckResult(
            name="torch",
            status="ok",
            message=f"TORCH_DEVICE={env_device} (PyTorch {torch.__version__}).",
        )
    device_ok = (
        (torch.cuda.is_available() and torch.cuda.device_count() > 0)
        or (hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.device_count() > 0)
        or torch.backends.mps.is_built() and torch.backends.mps.is_available()
    )
    if device_ok:
        return CheckResult(
            name="torch",
            status="ok",
            message="PyTorch detected at least one accelerator.",
        )
    return CheckResult(
        name="torch",
        status="warning",
        message="PyTorch import succeeded but no accelerators were detected.",
        remedy="Confirm drivers are available or set TORCH_DEVICE=cpu for CPU-only execution.",
    )


def _format_text(results: Iterable[CheckResult]) -> str:
    lines: list[str] = []
    for result in results:
        summary = f"[{result.status.upper():7}] {result.name}: {result.message}"
        lines.append(summary)
        if result.remedy:
            lines.append(f"          ↳ {result.remedy}")
    return "\n".join(lines)


def run_checks() -> list[CheckResult]:
    """Execute all diagnostic checks, returning structured results."""
    checks = [check_mpi, check_scheduler, check_wandb, check_torch_device]
    results: list[CheckResult] = []
    for check in checks:
        try:
            results.append(check())
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Diagnostic check %s crashed.", check.__name__)
            results.append(
                CheckResult(
                    name=check.__name__,
                    status="error",
                    message=f"Check raised {exc!r}",
                    remedy="Inspect the full stack trace above and report the failure.",
                )
            )
    return results


def parse_args(argv: Optional[Sequence[str]] = None):
    """Parse CLI arguments for the doctor command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect the current environment for ezpz launch readiness."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-friendly text.",
    )
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point used by the CLI glue."""
    args = parse_args(argv)
    results = run_checks()
    worst_status = max(results, key=lambda r: STATUS_PRIORITY[r.status]).status
    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        print(_format_text(results))
    return 0 if STATUS_PRIORITY[worst_status] < STATUS_PRIORITY["error"] else 1


def main() -> int:  # pragma: no cover - thin wrapper
    return run(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover - script execution
    raise SystemExit(main())
