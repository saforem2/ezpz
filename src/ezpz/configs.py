"""
ezpz/configs.py
"""
import os
import json
import logging
import subprocess
import rich.repr

from typing import Optional
from dataclasses import dataclass, asdict
from copy import deepcopy
from pathlib import Path
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

# -- Configure useful Paths -----------------------
# warnings.filterwarnings('ignore')
HERE = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = HERE.parent.parent
PROJECT_ROOT = PROJECT_DIR
CONF_DIR = HERE.joinpath('conf')
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


def load_ds_config(fpath: Optional[os.PathLike] = None) -> dict:
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
        return json.dumps(self.__dict__)

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
# @rich.repr.auto
class TrainConfig(BaseConfig):
    framework: str = "pytorch"
    backend: str = "DDP"
    use_wandb: bool = False
    seed: Optional[int] = None
    port: Optional[str] = None
    ds_config_path: Optional[os.PathLike] = None
    wandb_project_name: Optional[str] = None
    precision: Optional[str] = None
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
