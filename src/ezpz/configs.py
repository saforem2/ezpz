"""
ezpz/configs.py
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any, Optional, Sequence, Union

from omegaconf import DictConfig, OmegaConf
import rich.repr
from rich.syntax import Syntax
from rich.tree import Tree

log = logging.getLogger(__name__)

# -- Configure useful Paths -----------------------
# warnings.filterwarnings('ignore')
HERE = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = HERE.parent.parent
PROJECT_ROOT = PROJECT_DIR
CONF_DIR = HERE.joinpath('conf')
BIN_DIR = HERE.joinpath('bin')
SAVEJOBENV = BIN_DIR.joinpath('savejobenv')
GETJOBENV = BIN_DIR.joinpath('getjobenv')
DS_CONFIG_PATH = CONF_DIR.joinpath('ds_config.yaml')
LOGS_DIR = PROJECT_DIR.joinpath('logs')
OUTPUTS_DIR = HERE.joinpath('outputs')
QUARTO_OUTPUTS_DIR = PROJECT_DIR.joinpath('qmd', 'outputs')

CONF_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
QUARTO_OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
OUTDIRS_FILE = OUTPUTS_DIR.joinpath('outdirs.log')


FRAMEWORKS = {
    'pytorch': ['p', 'pt', 'torch', 'pytorch'],
    'tensorflow': ['t', 'tf', 'tflow', 'tensorflow'],
}
BACKENDS = {
    'pytorch': ['ddp', 'ds', 'dspeed', 'deepspeed', 'h', 'hvd', 'horovod'],
    'tensorflow': ['h', 'hvd', 'horovod']
}


def getjobenv():
    print(GETJOBENV)
    return GETJOBENV


def savejobenv():
    print(SAVEJOBENV)
    return SAVEJOBENV


def load_ds_config(
        fpath: Optional[Union[str, os.PathLike, Path]] = None
) -> dict:
    fpath = DS_CONFIG_PATH if fpath is None else fpath
    cfgpath = Path(fpath)
    log.info(
        'Loading DeepSpeed config from: '
        f'{cfgpath.resolve().as_posix()}'
    )
    if cfgpath.suffix == '.json':
        import json
        with cfgpath.open('r') as f:
            ds_config = json.load(f)
        return ds_config
    if cfgpath.suffix == '.yaml':
        import yaml
        with cfgpath.open('r') as stream:
            ds_config = dict(yaml.safe_load(stream))
        return ds_config
    raise TypeError('Unexpected FileType')


def command_exists(cmd: str) -> bool:
    result = subprocess.Popen(
        f'type {cmd}',
        stdout=subprocess.PIPE,
        shell=True
    )
    return result.wait() == 0


def git_ds_info():
    from deepspeed.env_report import main as ds_report
    ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    log.info(
        f'**** Git info for DeepSpeed:'
        f' git_hash={git_hash} git_branch={git_branch} ****'
    )


@dataclass
@rich.repr.auto
class BaseConfig(ABC):
    @abstractmethod
    def to_str(self) -> str:
        pass

    def to_json(self) -> str:
        return json.dumps(self.__dict__.items, indent=4)

    def get_config(self) -> dict:
        return asdict(self)

    def to_dict(self) -> dict:
        return deepcopy(self.__dict__)

    def to_file(self, fpath: os.PathLike) -> None:
        with Path(fpath).open('w') as f:
            json.dump(self.to_json(), f, indent=4)

    def from_file(self, fpath: os.PathLike) -> None:
        with Path(fpath).open('r') as f:
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
        return '_'.join([
            f'fw-{self.framework}',
            f'be-{self.backend}',
        ])

    def __post_init__(self):
        # assert self.framework.lower() in FRAMEWORKS.values()
        # if self.seed is None:
        #     self.seed = np.random.randint(0, 2**32 - 1)
        assert self.framework in [
            't', 'tf', 'tflow', 'tensorflow',
            'p', 'pt', 'ptorch', 'torch', 'pytorch',
        ]
        if self.framework in ["t", "tf", "tensorflow"]:
            assert self.backend.lower() in BACKENDS['tensorflow']
        else:
            assert self.backend.lower() in BACKENDS['pytorch']
        if self.use_wandb and self.wandb_project_name is None:
            self.wandb_project_name = os.environ.get(
                "WANDB_PROJECT",
                os.environ.get(
                    "WB_PROJECT",
                    "ezpz"
                )
            )
        if self.framework in ['p', 'pt', 'ptorch', 'torch', 'pytorch']:
            if self.backend.lower() in ['ds', 'deepspeed', 'dspeed']:
                self.ds_config = load_ds_config(
                    DS_CONFIG_PATH if self.ds_config_path is None
                    else self.ds_config_path
                )


def print_config_tree(
        cfg: DictConfig,
        resolve: bool = True,
        save_to_file: bool = True,
        verbose: bool = True,
        style: str = 'tree',
        print_order: Optional[Sequence[str]] = None,
        outfile: Optional[Union[str, os.PathLike, Path]] = None,
) -> Tree:
    """Prints the contents of a DictConfig as a tree structure using the Rich
    library.

    - cfg: A DictConfig composed by Hydra.
    - print_order: Determines in what order config components are printed.
    - resolve: Whether to resolve reference fields of DictConfig.
    - save_to_file: Whether to export config to the hydra output folder.
    """
    # from enrich.console import get_console
    from rich.console import Console
    # from enrich.console import get_width()
    from enrich.config import STYLES
    from rich.theme import Theme
    theme = Theme(STYLES)
    # from enrich.console import get_theme

    # console = get_console(record=True)  # , width=min(200))
    console = Console(record=True, theme=Theme(STYLES))
    style = "tree" if style is None else style
    tree = Tree("CONFIG", style=style, guide_style=style)
    queue = []
    # add fields from `print_order` to queue
    if print_order is not None:
        for field in print_order:
            queue.append(field) if field in cfg else log.warning(
                f"Field '{field}' not found in config. "
                f"Skipping '{field}' config printing..."
            )
    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)
    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
        branch.add(Syntax(branch_content, "yaml"))
    # print config tree
    if verbose or save_to_file:
        console.print(tree)
        if save_to_file:
            outfpath = (
                Path(os.getcwd()).joinpath('config_tree.log')
                if outfile is None else Path(outfile)
            )
            console.save_text(outfpath.as_posix())
            # with outfpath.open('w') as f:
            #     rich.print(tree, file=f)
        # console.print(tree)
        # rich.print(tree)
    # save config tree to file
    # if save_to_file:
    #     outfpath = (
    #         Path(os.getcwd()).joinpath('config_tree.log')
    #         if outfile is None else Path(outfile)
    #     )
    #     console.save_text(outfpath)
    #     with outfpath.open('w') as f:
    #         rich.print(tree, file=f)
    return tree
