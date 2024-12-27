"""
ezpz/configs.py
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import yaml
import logging
import os
import shutil
from pathlib import Path
import subprocess
from typing import Any, Callable, Optional, Sequence, Union
import numpy as np

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
import rich.repr
from rich.text import Text
from rich.tree import Tree

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
    os.environ.get(
        "PBS_O_WORKDIR",
        os.environ.get(
            "SLURM_SUBMIT_DIR",
            os.getcwd()
        )
    )
)
LOGS_DIR = WORKING_DIR.joinpath("logs")
OUTPUTS_DIR = WORKING_DIR.joinpath("outputs")
# OUTPUTS_DIR = HERE.joinpath("outputs")
QUARTO_OUTPUTS_DIR = PROJECT_DIR.joinpath("qmd", "outputs")

CONF_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
QUARTO_OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
OUTDIRS_FILE = OUTPUTS_DIR.joinpath("outdirs.log")


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


def getjobenv_dep():
    print(GETJOBENV)
    return GETJOBENV


def savejobenv_dep():
    print(SAVEJOBENV)
    return SAVEJOBENV


def get_timestamp(fstr: Optional[str] = None) -> str:
    """Get formatted timestamp."""
    import datetime

    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime("%Y-%m-%d-%H%M%S")
    return now.strftime(fstr)


def cmd_exists(cmd: str) -> bool:
    """Check whether command exists.

    >>> cmd_exists('ls')
    True
    >>> cmd_exists('hostname')
    True
    """
    return shutil.which(cmd) is not None


def get_scheduler() -> str:
    from ezpz import get_machine, get_hostname

    machine = get_machine(get_hostname())
    if machine.lower() in ["thetagpu", "sunspot", "polaris", "aurora"]:
        return SCHEDULERS["ALCF"]
    elif machine.lower() in ["frontier"]:
        return SCHEDULERS["OLCF"]
    elif machine.lower() in ["nersc", "perlmutter"]:
        return SCHEDULERS["NERSC"]
    else:
        return "LOCAL"
    # raise RuntimeError(f'Unknown {machine=}')


def load_ds_config(
    fpath: Optional[
        Union[str, os.PathLike, Path]
    ] = None,  # type:ignore[reportDeprecated]
) -> dict[str, Any]:
    from ezpz.configs import DS_CONFIG_PATH

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
    # import logging.config
    import yaml

    cfp = CONF_DIR.joinpath("hydra", "job_logging", "custom.yaml")
    with cfp.open("r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    import logging
    import logging.config

    logging.config.dictConfig(get_logging_config())
    log = logging.getLogger(name)
    if level is not None:
        log.setLevel("INFO")
    return log


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
    from ezpz.log.console import get_console
    from rich.json import JSON

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
        if not isinstance(json_str, str):
            raise TypeError(
                f"json must be str. Did you mean print_json(data={json_str!r}) ?"
            )
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
    from ezpz.log.handler import RichHandler as EnrichHandler
    from rich.logging import RichHandler

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
    result = subprocess.Popen(f"type {cmd}", stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


def git_ds_info():
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
    @abstractmethod
    def to_str(self) -> str:
        pass

    def to_json(self) -> str:
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
        return asdict(self)

    def to_dict(self) -> dict:
        return deepcopy(self.__dict__)

    def to_file(self, fpath: os.PathLike) -> None:
        with Path(fpath).open("w") as f:
            json.dump(self.to_json(), f, indent=4)

    def from_file(self, fpath: os.PathLike) -> None:
        with Path(fpath).open("r") as f:
            config = json.load(f)
        self.__init__(**config)

    def __getitem__(self, key):
        return super().__getattribute__(key)


@dataclass
@rich.repr.auto
class TrainConfig(BaseConfig):
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

    def to_str(self) -> str:
        return "_".join(
            [
                f"fw-{self.framework}",
                f"be-{self.backend}",
            ]
        )

    def __post_init__(self):
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
    from ezpz.log.config import STYLES
    from rich.theme import Theme

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
