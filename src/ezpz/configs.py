"""
ezpz/configs.py
"""

import json
import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path
from typing import (Any, Callable, Iterable, Optional, Protocol, Sequence,
                    Union, cast)

try:  # Python 3.10+
    from importlib.metadata import EntryPoint, entry_points
except ImportError:  # pragma: no cover - fallback for older runtimes
    from importlib_metadata import EntryPoint, entry_points  # type: ignore

import numpy as np
import rich.repr
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.text import Text
from rich.tree import Tree
# import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.utils.versions import require_version

log = logging.getLogger(__name__)

# -- Configure useful Paths -----------------------
# warnings.filterwarnings('ignore')
HERE = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = HERE.parent.parent
PROJECT_ROOT = PROJECT_DIR
CONF_DIR = HERE.joinpath("conf")
BIN_DIR = HERE.joinpath("bin")
SAVEJOBENV = BIN_DIR.joinpath("savejobenv")
UTILS = BIN_DIR.joinpath("utils.sh")
GETJOBENV = BIN_DIR.joinpath("getjobenv")
DS_CONFIG_PATH = CONF_DIR.joinpath("ds_config.yaml")
DS_CONFIG_YAML = CONF_DIR.joinpath("ds_config.yaml")
DS_CONFIG_JSON = CONF_DIR.joinpath("ds_config.json")
# LOGS_DIR = PROJECT_DIR.joinpath("logs")
WORKING_DIR = Path(
    os.environ.get("PBS_O_WORKDIR", os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()))
)
LOGS_DIR = WORKING_DIR.joinpath("logs")
OUTPUTS_DIR = WORKING_DIR.joinpath("outputs")
QUARTO_OUTPUTS_DIR = PROJECT_DIR.joinpath("qmd", "outputs")

# CONF_DIR.mkdir(exist_ok=True, parents=True)
# LOGS_DIR.mkdir(exist_ok=True, parents=True)
# QUARTO_OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
# OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
# OUTDIRS_FILE = OUTPUTS_DIR.joinpath('outdirs.log')


CAUSAL_LM_MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())  # type:ignore
CAUSAL_LM_MODEL_TYPES = tuple(
    getattr(conf, "model_type", "") for conf in CAUSAL_LM_MODEL_CONFIG_CLASSES
)


FRAMEWORKS = {
    "pytorch": ["p", "pt", "torch", "pytorch"],
    "tensorflow": ["t", "tf", "tflow", "tensorflow"],
}
BACKENDS = {
    "pytorch": ["ddp", "ds", "dspeed", "deepspeed", "h", "hvd", "horovod"],
    "tensorflow": ["h", "hvd", "horovod"],
}

SCHEDULERS = {
    "ALCF": "PBS",
    "OLCF": "SLURM",
    "NERSC": "SLURM",
    "LOCALHOST": "NONE",
}


PathLike = Union[str, os.PathLike, Path]
ScalarLike = Union[int, float, bool, np.floating]

SCHEDULER_ENTRYPOINT_GROUP = "ezpz.schedulers"


class SchedulerDetector(Protocol):
    """Callable signature expected for scheduler detector plug-ins."""

    def __call__(
        self,
        env: dict[str, str],
        *args: Any,
        **kwargs: Any,
    ) -> Optional[str]:
        ...


_ENTRY_POINT_PLUGINS: Optional[list[SchedulerDetector]] = None
_RUNTIME_PLUGINS: list[SchedulerDetector] = []


def getjobenv_dep():
    """Return the ``getjobenv`` helper path (for debugging)."""
    print(GETJOBENV)
    return GETJOBENV


def savejobenv_dep():
    """Return the ``savejobenv`` helper path (for debugging)."""
    print(SAVEJOBENV)
    return SAVEJOBENV


def cmd_exists(cmd: str) -> bool:
    """Check whether command exists.

    >>> cmd_exists("ls")
    True
    >>> cmd_exists("hostname")
    True
    """
    return shutil.which(cmd) is not None


def register_scheduler_plugin(detector: SchedulerDetector) -> None:
    """Register an in-process scheduler detector callable.

    The detector should accept a dictionary of environment variables and may
    optionally accept keyword arguments ``hostname`` and ``machine``.  It must
    return a scheduler identifier (e.g. ``"PBS"``) or ``None``.
    """

    _RUNTIME_PLUGINS.append(detector)


def _clear_scheduler_plugins() -> None:
    """Reset cached plug-ins (intended for testing)."""

    _RUNTIME_PLUGINS.clear()
    global _ENTRY_POINT_PLUGINS
    _ENTRY_POINT_PLUGINS = None


def _load_entry_point_plugins() -> list[SchedulerDetector]:
    """Return plug-ins discovered via importlib entry points."""

    global _ENTRY_POINT_PLUGINS
    if _ENTRY_POINT_PLUGINS is not None:
        return _ENTRY_POINT_PLUGINS

    plugins: list[SchedulerDetector] = []
    try:
        discovered: Any = entry_points()
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("Unable to inspect scheduler entry points: %s", exc)
        discovered = ()
    if hasattr(discovered, "select"):
        selector = getattr(discovered, "select")
        selected_iter = selector(group=SCHEDULER_ENTRYPOINT_GROUP)
        selected: Iterable[EntryPoint] = cast(Iterable[EntryPoint], selected_iter)
    elif isinstance(discovered, dict):
        selected = cast(
            Iterable[EntryPoint],
            discovered.get(SCHEDULER_ENTRYPOINT_GROUP, ()),
        )
    else:  # pragma: no cover - defensive
        selected = cast(Iterable[EntryPoint], discovered)

    for ep in selected:
        try:
            plugin = ep.load()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "Failed to load scheduler plug-in %s: %s",
                getattr(ep, "name", ep),
                exc,
            )
            continue
        if callable(plugin):
            plugins.append(cast(SchedulerDetector, plugin))
        else:
            log.warning(
                "Scheduler plug-in %s is not callable; ignoring",
                getattr(ep, "name", ep),
            )
    _ENTRY_POINT_PLUGINS = plugins
    return plugins


def _iter_scheduler_plugins() -> list[SchedulerDetector]:
    """Return all configured scheduler detector callables."""

    plugins: list[SchedulerDetector] = []
    plugins.extend(_load_entry_point_plugins())
    plugins.extend(_RUNTIME_PLUGINS)
    return plugins


def _call_scheduler_plugin(
    detector: SchedulerDetector,
    env: dict[str, str],
    *,
    hostname: Optional[str] = None,
    machine: Optional[str] = None,
) -> Optional[str]:
    """Invoke a scheduler detector with best-effort argument matching."""

    try:
        return detector(env, hostname=hostname, machine=machine)
    except TypeError:
        try:
            return detector(env, hostname=hostname)
        except TypeError:
            try:
                return detector(env)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Scheduler plug-in %r failed: %s", detector, exc)
                return None
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Scheduler plug-in %r failed: %s", detector, exc)
        return None


def _detect_scheduler_via_plugins(
    env: dict[str, str],
    hostname: str,
    machine: str,
) -> Optional[str]:
    """Run plug-ins until one returns a scheduler string."""

    for detector in _iter_scheduler_plugins():
        candidate = _call_scheduler_plugin(
            detector, env, hostname=hostname, machine=machine
        )
        if candidate is None:
            continue
        normalized = candidate.strip()
        if normalized:
            return normalized.upper()
    return None


def get_scheduler() -> str:
    """Infer the active scheduler from environment variables or hostname."""
    from ezpz import get_hostname, get_machine

    env = dict(os.environ)
    hostname = get_hostname()
    machine = get_machine(hostname)

    plugin_result = _detect_scheduler_via_plugins(env, hostname, machine)
    if plugin_result is not None:
        return plugin_result

    if env.get("PBS_JOBID"):
        return "PBS"
    if env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID"):
        return "SLURM"

    if machine.lower() in [
        "thetagpu",
        "sunspot",
        "polaris",
        "aurora",
        "sophia",
    ]:
        return SCHEDULERS["ALCF"]
    if machine.lower() in ["frontier"]:
        return SCHEDULERS["OLCF"]
    if machine.lower() in ["nersc", "perlmutter"]:
        return SCHEDULERS["NERSC"]
    return "UNKNOWN"
    # raise RuntimeError(f'Unknown {machine=}')


def load_ds_config(
    fpath: Optional[
        Union[str, os.PathLike, Path]
    ] = None,  # type:ignore[reportDeprecated]
) -> dict[str, Any]:
    """Load a DeepSpeed configuration file (JSON or YAML)."""
    fpath = Path(DS_CONFIG_PATH) if fpath is None else f"{fpath}"
    cfgpath = Path(fpath)
    if cfgpath.suffix == ".json":
        with cfgpath.open("r") as f:
            ds_config: dict[str, Any] = json.load(f)
        return ds_config
    if cfgpath.suffix == ".yaml":
        with cfgpath.open("r") as stream:
            dsconfig: dict[str, Any] = dict(yaml.safe_load(stream))
        return dsconfig
    raise TypeError("Unexpected FileType")


def get_logging_config() -> dict:
    """Return the logging configuration dictionary used by ``logging.config``."""
    # import logging.config
    import yaml

    cfp = CONF_DIR.joinpath("hydra", "job_logging", "custom.yaml")
    with cfp.open("r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    config.setdefault("loggers", {})
    return config


# def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
#     import logging
#     import logging.config
#
#     logging.config.dictConfig(get_logging_config())
#     log = logging.getLogger(name)
#     if level is not None:
#         log.setLevel("INFO")
#     return log


def print_json(
    json_str: Optional[str] = None,
    console: Optional[Console] = None,
    *,
    data: Any = None,
    indent: Union[None, int, str] = 2,
    highlight: bool = True,
    skip_keys: bool = False,
    ensure_ascii: bool = False,
    check_circular: bool = True,
    allow_nan: bool = True,
    default: Optional[Callable[[Any], Any]] = None,
    sort_keys: bool = False,
) -> None:
    """Pretty prints JSON. Output will be valid JSON.

    Args:
        json_str (Optional[str]): A string containing JSON.
        data (Any): If json is not supplied, then encode this data.
        indent (Union[None, int, str], optional): Number of spaces to indent.
            Defaults to 2.
        highlight (bool, optional): Enable highlighting of output:
            Defaults to True.
        skip_keys (bool, optional): Skip keys not of a basic type.
            Defaults to False.
        ensure_ascii (bool, optional): Escape all non-ascii characters.
            Defaults to False.
        check_circular (bool, optional): Check for circular references.
            Defaults to True.
        allow_nan (bool, optional): Allow NaN and Infinity values.
            Defaults to True.
        default (Callable, optional): A callable that converts values
            that can not be encoded in to something that can be JSON
            encoded.
            Defaults to None.
        sort_keys (bool, optional): Sort dictionary keys. Defaults to False.
    """
    if json_str is None and data is None:
        raise ValueError(
            "Either `json_str` or `data` must be provided. "
            "Did you mean print_json(data={data!r}) ?"
        )
    if json_str is not None and data is not None:
        raise ValueError(
            " ".join(
                [
                    "Only one of `json_str` or `data` should be provided.",
                    "Did you mean print_json(json_str={json_str!r}) ?",
                    "Or print_json(data={data!r}) ?",
                    "Received both:",
                    f"json_str={json_str!r}",
                    f"data={data!r}",
                ]
            )
        )
    from rich.json import JSON

    from ezpz.log.console import get_console

    console = get_console() if console is None else console
    if json_str is None:
        json_renderable = JSON.from_data(
            data,
            indent=indent,
            highlight=highlight,
            skip_keys=skip_keys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            default=default,
            sort_keys=sort_keys,
        )
    else:
        json_renderable = JSON(
            json_str,
            indent=indent,
            highlight=highlight,
            skip_keys=skip_keys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            default=default,
            sort_keys=sort_keys,
        )
    assert console is not None and isinstance(console, Console)
    log.info(Text(str(json_renderable)).render(console=console))


def print_config(cfg: Union[dict, str]) -> None:
    """Render ``cfg`` to the active rich console."""
    # try:
    #     from hydra.utils import instantiate
    #     config = instantiate(cfg)
    # except Exception:
    #     config = OmegaConf.to_container(cfg, resolve=True)
    #     config = OmegaConf.to_container(cfg, resolve=True)
    # if isinstance(cfg, dict):
    #     jstr = json.dumps(cfg, indent=4)
    # else:
    #     jstr = cfg
    from rich.logging import RichHandler

    from ezpz.log.handler import RichHandler as EnrichHandler

    console = None
    for handler in log.handlers:
        if isinstance(handler, (RichHandler, EnrichHandler)):
            console = handler.console
    if console is None:
        from ezpz.log.console import get_console

        console = get_console()
    # console.print_json(data=cfg, indent=4, highlight=True)
    print_json(data=cfg, console=console, indent=4, highlight=True)


def command_exists(cmd: str) -> bool:
    """Return ``True`` if ``cmd`` is available on ``PATH``."""
    return shutil.which(cmd) is not None


def git_ds_info():
    """Log the output of DeepSpeed's environment report plus Git metadata."""
    from deepspeed.env_report import main as ds_report

    ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists("git"):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode("utf-8").strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode("utf-8").strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    log.info(
        f"**** Git info for DeepSpeed:"
        f" git_hash={git_hash} git_branch={git_branch} ****"
    )


@dataclass
class BaseConfig(ABC):
    """Common utilities shared by ezpz configuration dataclasses."""

    @abstractmethod
    def to_str(self) -> str:
        """Return a short string that uniquely describes the configuration."""
        pass

    def to_json(self) -> str:
        """Return a JSON string representation of the configuration."""
        # name = (
        #     f'{name=}' if name is not None
        #     else f'{self.__class__.__name__}'
        # )
        return json.dumps(deepcopy(self.__dict__), indent=4)
        # return '\n'.join(
        #     [
        #         (f'{name=}' if name is not None
        #          else f'{self.__class__.__name__}'),
        #         json.dumps(deepcopy(self.__dict__), indent=4),
        #     ]
        # )

    def get_config(self) -> dict:
        """Return the configuration as a standard dictionary via ``asdict``."""
        return asdict(self)

    def to_dict(self) -> dict:
        """Return a deep copy of the dataclass ``__dict__``."""
        return deepcopy(self.__dict__)

    def to_file(self, fpath: os.PathLike) -> None:
        """Write the configuration to ``fpath`` in JSON format."""
        with Path(fpath).open("w") as f:
            json.dump(self.to_json(), f, indent=4)

    def from_file(self, fpath: os.PathLike) -> None:
        """Populate the configuration from a JSON file."""
        with Path(fpath).open("r") as f:
            config = json.load(f)
        self.__init__(**config)

    def __getitem__(self, key):
        """Provide dictionary-style indexing for convenience."""
        return super().__getattribute__(key)


@dataclass(init=False)
@rich.repr.auto
class TrainConfig(BaseConfig):
    """High-level training options shared by ezpz scripts."""

    gas: int = 1
    # ---- [NOTE]+ Framework + Backend ----------------
    # `framework`: `{'backend'}`
    #   • `tensorflow`: `{'horovod'}`
    #   • `pytorch`: `{'DDP', 'deepspeed', 'horovod'}`
    # -------------------------------------------------
    framework: str = "pytorch"
    backend: str = "DDP"
    use_wandb: bool = False
    seed: Optional[int] = None
    port: Optional[str] = None
    dtype: Optional[Any] = None
    load_from: Optional[str] = None
    save_to: Optional[str] = None
    ds_config_path: Optional[str] = None
    wandb_project_name: Optional[str] = None
    ngpus: Optional[int] = None
    extras: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __init__(self, **kwargs: Any) -> None:
        """Populate known fields while capturing any extras in ``self.extras``."""
        extras: dict[str, Any] = {}
        for name, field_def in self.__dataclass_fields__.items():
            if name == "extras":
                continue
            if name in kwargs:
                value = kwargs.pop(name)
            elif field_def.default is not MISSING:
                value = field_def.default
            elif field_def.default_factory is not MISSING:  # type: ignore[attr-defined]
                value = field_def.default_factory()  # type: ignore[misc]
            else:
                value = None
            setattr(self, name, value)

        for key, value in kwargs.items():
            setattr(self, key, value)
            extras[key] = value
        self.extras = extras

    def to_str(self) -> str:
        """Return a compact identifier combining framework and backend."""
        return "_".join(
            [
                f"fw-{self.framework}",
                f"be-{self.backend}",
            ]
        )

    def __post_init__(self):
        """Validate framework/backend compatibility after initialisation."""
        # assert self.framework.lower() in FRAMEWORKS.values()
        # if self.seed is None:
        #     self.seed = np.random.randint(0, 2**32 - 1)
        assert self.framework in [
            "t",
            "tf",
            "tflow",
            "tensorflow",
            "p",
            "pt",
            "ptorch",
            "torch",
            "pytorch",
        ]
        if self.framework in ["t", "tf", "tensorflow"]:
            assert self.backend.lower() in BACKENDS["tensorflow"]
        else:
            assert self.backend.lower() in BACKENDS["pytorch"]
        if self.use_wandb and self.wandb_project_name is None:
            self.wandb_project_name = os.environ.get(
                "WANDB_PROJECT", os.environ.get("WB_PROJECT", "ezpz")
            )
        if self.framework in ["p", "pt", "ptorch", "torch", "pytorch"]:
            if self.backend.lower() in ["ds", "deepspeed", "dspeed"]:
                self.ds_config = load_ds_config(
                    DS_CONFIG_PATH
                    if self.ds_config_path is None
                    else self.ds_config_path
                )


@dataclass
class ZeroConfig:
    """Subset of DeepSpeed ZeRO options exposed via the ezpz CLI."""

    stage: int = 0
    allgather_partitions: Optional[bool] = None
    allgather_bucket_size: int = int(5e8)
    overlap_comm: Optional[bool] = None
    reduce_scatter: bool = True
    reduce_bucket_size: int = int(5e8)
    contiguous_gradients: Optional[bool] = None
    offload_param: Optional[dict] = None
    offload_optimizer: Optional[dict] = None
    stage3_max_live_parameters: int = int(1e9)
    stage3_max_reuse_distance: int = int(1e9)
    stage3_prefetch_bucket_size: int = int(5e8)
    stage3_param_persistence_threshold: int = int(1e6)
    sub_group_size: Optional[int] = None
    elastic_checkpoint: Optional[dict] = None
    stage3_gather_16bit_weights_on_model_save: Optional[bool] = None
    ignore_unused_parameters: Optional[bool] = None
    round_robin_gradients: Optional[bool] = None
    zero_hpz_partition_size: Optional[int] = None
    zero_quantized_weights: Optional[bool] = None
    zero_quantized_gradients: Optional[bool] = None
    log_trace_cache_warnings: Optional[bool] = None


@dataclass
class HfModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    wandb_project_name: Optional[str] = field(  # type:ignore
        default=None,
        metadata={
            "help": (
                "The name of the wandb project to use. If not specified, will use the model name."
            )
        },
    )

    model_name_or_path: Optional[str] = field(  # type:ignore
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str | None] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(CAUSAL_LM_MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        """Validate mutually exclusive Hugging Face model configuration options."""
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class HfDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data."},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_split_str: Optional[str] = field(
        default=None,
        metadata={
            "help": "The split string to use for the train split (via the datasets library)."
        },
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the train split to use (via the datasets library)."
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation split to use (via the datasets library)."
        },
    )
    validation_split_str: Optional[str] = field(
        default=None,
        metadata={
            "help": "The split string to use for the validation split (via the datasets library)."
        },
    )
    test_split_name: Optional[str] = field(
        default="test",
        metadata={
            "help": "The name of the test split to use (via the datasets library)."
        },
    )
    test_split_str: Optional[str] = field(
        default=None,
        metadata={
            "help": "The split string to use for the test split (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )

    def __post_init__(self):
        """Validate dataset arguments and ensure required files are present."""
        if self.streaming:
            require_version(
                "datasets>=2.0.0",
                "The streaming feature requires `datasets>=2.0.0`",
            )

        if (
            self.dataset_name is None
            and self.data_path is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "You must specify at least one of the following: "
                "a dataset name, a data path, a training file, or a validation file."
            )

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`train_file` should be a csv, a json or a txt file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`validation_file` should be a csv, a json or a txt file."


def print_config_tree(
    cfg: DictConfig,
    resolve: bool = True,
    save_to_file: bool = True,
    verbose: bool = True,
    style: str = "tree",
    print_order: Optional[Sequence[str]] = None,
    highlight: bool = True,
    outfile: Optional[Union[str, os.PathLike, Path]] = None,
) -> Tree:
    """Prints the contents of a DictConfig as a tree structure using the Rich
    library.

    - cfg: A DictConfig composed by Hydra.
    - print_order: Determines in what order config components are printed.
    - resolve: Whether to resolve reference fields of DictConfig.
    - save_to_file: Whether to export config to the hydra output folder.
    """
    from rich.console import Console
    from rich.theme import Theme

    from ezpz.log.config import STYLES

    name = cfg.get("_target_", "cfg")
    console = Console(record=True, theme=Theme(STYLES))
    tree = Tree(label=name, highlight=highlight)
    queue = []
    # add fields from `print_order` to queue
    if print_order is not None:
        for field in print_order:
            (
                queue.append(field)
                if field in cfg
                else log.warning(
                    f"Field '{field}' not found in config. "
                    f"Skipping '{field}' config printing..."
                )
            )
    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)
    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, highlight=highlight)  # , guide_style=style)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = str(OmegaConf.to_yaml(config_group, resolve=resolve))
            branch.add(Text(branch_content, style="red"))
        else:
            branch_content = str(config_group)
            branch.add(Text(branch_content, style="blue"))
    if verbose or save_to_file:
        console.print(tree)
        if save_to_file:
            outfpath = (
                Path(os.getcwd()).joinpath("config_tree.log")
                if outfile is None
                else Path(outfile)
            )
            console.save_text(outfpath.as_posix())
    return tree


@dataclass
class ViTConfig:
    img_size: int = 224
    patch_size: int = 16
    depth: int = 12
    num_heads: int = 12
    hidden_dim: int = 768
    mlp_dim: int = 3072
    dropout: float = 0.0
    attention_dropout: float = 0.0
    num_classes: int = 10

    def __post_init__(self):
        self.seq_len = (self.img_size // self.patch_size) ** 2  # 196, default


@dataclass
class timmViTConfig:
    img_size: int = 224
    batch_size: int = 128
    num_heads: int = 16
    head_dim: int = 64
    depth: int = 24
    patch_size: int = 16
    hidden_dim: int = 1024
    mlp_dim: int = 4096
    dropout: float = 0.0
    attention_dropout: float = 0.0
    num_classes: int = 1000

    def __post_init__(self):
        self.seq_len = (self.img_size // self.patch_size) ** 2  # 196, default


@dataclass
class TrainArgs:
    img_size: int
    batch_size: int
    num_heads: int
    head_dim: int
    depth: int
    patch_size: int
    dtype: str
    compile: bool
    attn_type: str
    num_workers: int
    max_iters: int
    format: Optional[str] = field(default_factory=str)
    cuda_sdpa_backend: Optional[str] = field(default_factory=str)
