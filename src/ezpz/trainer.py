"""
wordplay/trainer.py

```markdown
> [!NOTE]
>  If your cluster does not have Infiniband interconnect, prepend:
>  `NCCL_IB_DISABLE=1`
```
"""

from __future__ import absolute_import, annotations, division, print_function
from dataclasses import asdict
import logging
import math
from os import PathLike
import os
from pathlib import Path
import time
from typing import Any, Optional, Union
from omegaconf import DictConfig

import ezpz

from ezpz import (
    get_rank,
    get_torch_device,
    get_world_size,
    timeitlogit,
    get_local_rank,
)

# from ezpz.history import BaseHistory
from torch import optim
from torch import nn
import numpy as np
from rich.text import Text
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
import wandb

from wordplay.configs import ExperimentConfig, ModelConfig, add_to_ckpts_file
from wordplay.model import GPT

GradScaler = None
if torch.cuda.is_available():
    from torch.cuda.amp import GradScaler


log = logging.getLogger(__name__)

LOCAL_RANK = get_local_rank()
RANK = get_rank()
WORLD_SIZE = get_world_size(in_use=True)
DEVICE = get_torch_device()  # 'cuda' if torch.cuda.is_available() else 'cpu'


ScalarLike = Union[float, int, np.floating, bool]


def format_pair(k: str, v: ScalarLike) -> str:
    if isinstance(v, (int, bool, np.integer)):
        # return f'{k}={v:<3}'
        return f'{k}={v}'
    # return f'{k}={v:<3.4f}'
    return f'{k}={v:<6.4f}'


def summarize_dict(d: dict) -> str:
    return ' '.join([format_pair(k, v) for k, v in d.items()])


def grab_tensor(x: Any) -> np.ndarray | ScalarLike | None:
    if x is None:
        return None
    if isinstance(x, (int, float, bool, np.floating)):
        return x
    if isinstance(x, list):
        if isinstance(x[0], torch.Tensor):
            return grab_tensor(torch.stack(x))
        elif isinstance(x[0], np.ndarray):
            return np.stack(x)
        else:
            import tensorflow as tf

            if isinstance(x[0], tf.Tensor):
                return grab_tensor(tf.stack(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif callable(getattr(x, 'numpy', None)):
        assert callable(getattr(x, 'numpy'))
        return x.numpy()
    raise ValueError


def _average(val):
    if isinstance(val, (list, tuple)):
        if isinstance(val[0], torch.Tensor):
            val = grab_tensor(torch.stack(val))
        elif isinstance(val, np.ndarray):
            val = np.stack(val)
        else:
            val = val
    if isinstance(val, torch.Tensor):
        val = grab_tensor(val)

    if isinstance(val, (float, int, bool, np.floating, np.integer)):
        return val
    try:
        avg = np.mean(val).real  # type: ignore
        assert isinstance(avg, np.floating)
        return avg
    except Exception:
        log.exception(f'Failed to average {val}')
        log.warning('Returning val as is')
        return val


def average_dict(d: dict) -> dict:
    avgs = {}
    avg = 0.0
    for key, val in d.items():
        if val is None:
            continue
        if isinstance(val, dict):
            for k, v in val.items():
                kk = f'{key}/{k}'
                avg = _average(v)
                avgs[kk] = avg
        else:
            avg = _average(val)
            avgs[key] = avg
    return avgs


def allreduce_sum(x: torch.Tensor) -> float:
    torch.distributed.all_reduce(x, op=torch.distributed.SUM)
    return x / WORLD_SIZE


class TrainerMNIST:
    def __init__(self, cfg: DictConfig, scaler: Optional[GradScaler] = None):
        self.cfg = cfg
        # self.rank = RANK
        # if scaler is None:
        #     self.scaler = None

        # self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        # self.backend = self.cfg.backend
        # if WITH_DDP:
        #     init_process_group(RANK, SIZE, backend=self.backend)

        # self.setup_torch()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.world_size = get_world_size(in_use=True)
        self.data = self.setup_data()
        self.model = self.build_model()
        self.device_type = get_torch_device()
        self.device_id = f'{self.device_type}:{self.local_rank}'
        # if self.device == 'gpu':
        #     self.model.cuda()
        self.model.to(self.device_type)
        if self.world_size > 1:
            # self.device_id = f
            self.model = DDP(
                self.model,
                device_ids=[self.device_id],
            )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.build_optimizer(self.model)
        # if WITH_CUDA:
        #    self.loss_fn = self.loss_fn.cuda()

    def build_model(self) -> nn.Module:
        from ezpz.model import MnistCNN

        return MnistCNN()

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        # DDP: scale learning rate by the number of GPUs
        return optim.Adam(
            model.parameters(),
            lr=(self.world_size * self.cfg.lr_init),
        )

    def setup_torch(self):
        torch.manual_seed(self.cfg.seed)
        if self.device == 'gpu':
            # DDP: pin GPU to local rank
            torch.cuda.set_device(int(self.local_rank))
            torch.cuda.manual_seed(self.cfg.seed)

        if (
            self.cfg.num_threads is not None
            and isinstance(self.cfg.num_threads, int)
            and self.cfg.num_threads > 0
        ):
            torch.set_num_threads(self.cfg.num_threads)

        if RANK == 0:
            log.info(
                '\n'.join(
                    [
                        'Torch Thread Setup:',
                        f' Number of threads: {torch.get_num_threads()}',
                    ]
                )
            )

    def setup_data(self):
        # kwargs = {}
        # if self.device == 'gpu':
        from torchvision import datasets, transforms

        #     kwargs = {'num_workers': 1, 'pin_memory': True}
        train_dataset = datasets.MNIST(
            self.cfg.data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        # DDP: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            # **kwargs
        )
        test_dataset = datasets.MNIST(
            self.cfg.data_dir,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        # DDP: use DistributedSampler to partition the test data
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=self.world_size, rank=self.rank
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.cfg.batch_size
        )
        return {
            'train': {
                'sampler': train_sampler,
                'loader': train_loader,
            },
            'test': {
                'sampler': test_sampler,
                'loader': test_loader,
            },
        }

    def train_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # if WITH_CUDA:
        #     data, target = data.cuda(), target.cuda()
        data = data.to(self.device)
        self.optimizer.zero_grad()
        probs = self.model(data)
        loss = self.loss_fn(probs, target)
        if self.scaler is not None and isinstance(self.scaler, GradScaler):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        _, pred = probs.data.max(1)
        acc = (pred == target).sum()
        return loss, acc

    def train_epoch(
        self,
        epoch: int,
    ) -> dict:
        self.model.train()
        start = time.time()
        running_acc = torch.tensor(0.0).to(self.device_id)
        running_loss = torch.tensor(0.0).to(self.device_id)
        # if self.device_type == 'cuda':
        #     running_acc = running_acc.cuda()
        #     running_loss = running_loss.cuda()

        train_sampler = self.data['train']['sampler']
        train_loader = self.data['train']['loader']
        # DDP: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
        for bidx, (data, target) in enumerate(train_loader):
            loss, acc = self.train_step(data, target)
            running_acc += acc
            running_loss += loss.item()
            if bidx % self.cfg.logfreq == 0 and RANK == 0:
                # DDP: use train_sampler to determine the number of
                # examples in this workers partition
                metrics = {
                    'epoch': epoch,
                    'dt': time.time() - start,
                    'batch_acc': acc.item() / self.cfg.batch_size,
                    'batch_loss': loss.item() / self.cfg.batch_size,
                    'acc': running_acc / len(self.data['train']['sampler']),
                    'running_loss': (
                        running_loss / len(self.data['train']['sampler'])
                    ),
                }
                pre = [
                    f'[{RANK}]',
                    (  # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.cfg.epochs}:'
                        f' {bidx * len(data)}/{len(train_sampler)}'
                        f' ({100.0 * bidx / len(train_loader):.0f}%)]'
                    ),
                ]
                log.info(
                    ' '.join(
                        [*pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]]
                    )
                )

        running_loss = running_loss / len(train_sampler)
        running_acc = running_acc / len(train_sampler)
        training_acc = allreduce_sum(running_acc)
        loss_avg = allreduce_sum(running_loss)
        return {'loss': loss_avg, 'acc': training_acc}

    def test(self) -> float:
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.data['test']['loader']:
                if self.device == 'gpu':
                    data, target = data.cuda(), target.cuda()

                probs = self.model(data)
                _, predicted = probs.data.max(1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return correct / total


class TrainerLLM:
    def __init__(self, config: ExperimentConfig):
        # self.console = get_console()
        self.config = config
        self.ckpt = None
        self.rank = RANK
        self.world_size = WORLD_SIZE
        self.device = DEVICE
        # assert self.device == self.config.device_type
        # NOTE: ---------------------------------------------------------
        # config.optimizer.gas = (
        #     1 if config.optimizer.gradient_accumulation_steps is None
        #     else config.optimizer.gradient_accumulation_steps
        # ) -------------------------------------------------------------
        self.train_history = ezpz.History()
        self._gas = self.config.optimizer.gas
        self._lr = self.config.optimizer.learning_rate
        self._min_lr = self.config.optimizer.min_lr
        self._diters = self.config.optimizer.lr_decay_iters
        self._witers = self.config.train.warmup_iters
        if self.config.train.init_from == 'scratch':
            log.info('Initializing a new model from scratch')
            model = GPT(self.config.model)
        elif self.config.train.init_from == 'resume':
            model, ckpt = self.restore_from_ckpt()
            self.ckpt = ckpt
            self.config.set_iter_num(ckpt.get('iter_num', 0))
            self.config.set_best_val_loss(ckpt.get('best_val_loss', 1e9))
        elif self.config.train.init_from.startswith('gpt2'):
            model = self._init_gpt2()
        else:
            raise ValueError(
                f'Unexpected `init_from` = {self.config.train.init_from}. '
                'Exiting!'
            )
        # model = model
        # if torch.cuda.is_available():
        #     model.cuda()
        model.to(self.device)
        assert isinstance(model, GPT)
        assert issubclass(GPT, torch.nn.Module)
        num_params = model.get_num_params()
        if wandb.run is not None:
            wandb.run.config['num_params'] = num_params
        # model_block_size = int(self.model.config.block_size)
        if self.config.model.block_size < model.config.block_size:
            model.crop_block_size(self.config.model.block_size)
            self.config.model.set_block_size(self.config.model.block_size)
        optimizer = model.configure_optimizers(
            weight_decay=self.config.optimizer.weight_decay,
            learning_rate=self.config.optimizer.learning_rate,
            betas=(
                self.config.optimizer.beta1,
                self.config.optimizer.beta2,
            ),
            device_type=self.config.device_type,
        )
        if self.config.train.init_from == 'resume':
            assert (
                self.ckpt is not None
                and isinstance(self.ckpt, dict)
                and 'optimizer' in self.ckpt
            )
            optimizer.load_state_dict(self.ckpt['optimizer'])
            self.ckpt = None  # free up memory
        if self.config.train.compile:
            # unoptimized_model = self.model
            model = torch.compile(model)  # type:ignore
        # if WORLD_SIZE > 1:
        grad_scaler = None
        if self.config.train.backend.lower() == 'ddp':
            if torch.cuda.is_available():
                from torch.cuda.amp.grad_scaler import GradScaler

                grad_scaler = GradScaler(
                    enabled=(self.config.train.dtype == 'float16')
                )
            # self.optimizer = optimizer
            assert isinstance(model, torch.nn.Module)
            device = get_torch_device()
            local_rank = get_local_rank()
            devid = f'{device}:{local_rank}'
            log.critical(f'"{devid=}"')
            model.to(devid)
            model_engine = DDP(model, device_ids=[devid])
        elif self.config.train.backend.lower() in ['deepspeed', 'ds']:
            from ezpz import load_ds_config

            grad_scaler = None
            ds_config_path = self.config.train.ds_config_path
            if ds_config_path is None:
                from wordplay.configs import DS_CONFIG_PATH

                ds_config_path = DS_CONFIG_PATH
            self.ds_config = load_ds_config(ds_config_path)
            if 'optimizer' in self.ds_config.keys():
                optimizer = None
            assert isinstance(model, torch.nn.Module)
            ds_out = self._setup_deepspeed(
                ds_config=self.ds_config, model=model, optimizer=optimizer
            )
            model_engine = ds_out['model_engine']
            optimizer = ds_out['optimizer']
        else:
            raise ValueError(f'Unexpected {self.config.train.backend=}')
        self.model = model
        self.grad_scaler = grad_scaler
        self.model_engine = model_engine
        self.optimizer = optimizer

    def _init_gpt2(self) -> GPT:
        log.info(
            f'Initializing from OpenAI GPT-2 Weights: '
            f'{self.config.train.init_from}'
        )
        override_args = {'dropout': self.config.model.dropout}
        model = GPT.from_pretrained(self.config.train.init_from, override_args)
        model_cfg = {
            k: getattr(model.config, k)
            for k in [
                'n_layer',
                'n_head',
                'n_embd',
                'block_size',
                'bias',
                'vocab_size',
            ]
        }
        self.config.reset_model_config(ModelConfig(**model_cfg))
        return model

    def _setup_deepspeed(
        self,
        model: Optional[torch.nn.Module | GPT],
        ds_config: Optional[dict] = None,
        ds_config_path: Optional[os.PathLike] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict:
        """Setup DeepSpeed.

        TODO:
            - [ ] Deal with / fix gradient accumulation logic in `train_step`
            - [ ] Test / generalize optimizer creation
        """
        import deepspeed
        from ezpz import load_ds_config

        if ds_config is None:
            assert ds_config_path is not None, (
                'One of `ds_config` or `ds_config_path` must be specified.'
            )
            ds_config = load_ds_config(Path(ds_config_path).as_posix())
        assert ds_config is not None
        if self.config.train.wandb_project is not None:
            ds_config['wandb'].update(
                {
                    'enabled': True,
                    'project': self.config.train.wandb_project,
                }
            )
        log.warning(
            f'Setting `train_micro_batch_size_per_gpu` to '
            f'{self.config.model.batch_size=}'
        )
        ds_config.update(
            {'train_micro_batch_size_per_gpu': self.config.model.batch_size}
        )
        assert model is not None and (
            isinstance(model, (torch.nn.Module, GPT))
            or issubclass(model, torch.nn.Module)
        )
        assert model is not None
        if optimizer is not None and isinstance(
            optimizer, torch.optim.Optimizer
        ):
            engine, optimizer, *_ = deepspeed.initialize(
                model=model,
                config=ds_config,
                optimizer=optimizer,
            )
        elif 'optimizer' in ds_config.keys():
            engine, optimizer, *_ = deepspeed.initialize(
                model=model,
                config=ds_config,
                model_parameters=model.parameters(),
            )
        else:
            raise ValueError('Unable to initialize DeepSpeed')
        assert engine is not None and optimizer is not None
        return {
            'model_engine': engine,
            'optimizer': optimizer,
            'ds_config': ds_config,
        }

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        # data = self.config.train_data if split == 'train'
        # else self.config.val_data
        data = self.config.data.data.get(split, None)
        assert data is not None
        ix = torch.randint(
            len(data) - self.config.model.block_size,
            (self.config.model.batch_size,),
        )
        block_size = self.config.model.block_size
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        if self.config.device_type == 'cuda':
            x = x.pin_memory().to(self.config.device_type, non_blocking=True)
            y = y.pin_memory().to(self.config.device_type, non_blocking=True)
        else:
            x = x.to(self.config.device_type)
            y = y.to(self.config.device_type)
        return x, y

    def get_lr(self, it: int) -> float:
        if it < self._witers:
            return self._lr * it / self._witers
        if it > self._diters:
            return self._min_lr
        decay_ratio = (it - self._witers) / (self._diters - self._witers)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self._min_lr + coeff * (self._lr - self._min_lr)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in self.config.data.data.keys():
            losses = torch.zeros(self.config.train.eval_iters)
            for k in range(self.config.train.eval_iters):
                x, y = self.get_batch(split)
                with self.config.ctx:
                    _, loss = self.model_engine(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def restore_from_ckpt(
        self, ckpt_dir: Optional[str | PathLike] = None
    ) -> tuple[torch.nn.Module, dict]:
        log.info(f'Resuming training from {self.config.data.out_dir}')
        ckpt_dir = (
            str(self.config.data.out_dir) if ckpt_dir is None else ckpt_dir
        )
        assert ckpt_dir is not None
        ckpt_path = Path(ckpt_dir).joinpath('ckpt.pt')
        checkpoint = torch.load(
            ckpt_path, map_location=self.config.train.device
        )
        ckpt_model = checkpoint['model_args']
        model_config = ModelConfig(
            n_layer=ckpt_model['n_layer'],
            n_head=ckpt_model['n_head'],
            n_embd=ckpt_model['n_embd'],
            block_size=ckpt_model['block_size'],
            bias=ckpt_model['bias'],
            vocab_size=ckpt_model['vocab_size'],
        )
        model = GPT(model_config)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model, checkpoint

    def _forward_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        t0 = time.perf_counter()
        with self.config.ctx:
            logits, loss = self.model_engine(x, y)
        return {'logits': logits, 'loss': loss, 'dt': time.perf_counter() - t0}

    def _backward_step(
        self,
        loss: torch.Tensor,
        propagate_grads: bool = False,
    ) -> float:
        t0 = time.perf_counter()
        if self.config.train.backend.lower() in ['ds', 'deepspeed']:
            self.model_engine.backward(loss)  # type:ignore
            self.model_engine.step(loss)  # type:ignore
        else:
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()  # type:ignore
            if propagate_grads:
                if self.config.optimizer.grad_clip != 0.0:
                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(  # pyright: ignore
                        self.model_engine.parameters(),
                        self.config.optimizer.grad_clip,
                    )
                if self.grad_scaler is not None:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

        return time.perf_counter() - t0

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict:
        lr = (
            self.get_lr(self.config.iter_num)
            if self.config.optimizer.decay_lr
            else self._lr
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        dtf = []
        dtb = []
        dt = []
        loss = torch.tensor(0.0)
        for micro_step in range(self._gas):
            is_last_micro_step = micro_step == self._gas - 1
            # NOTE: -----------------------------------------------------------
            # In DDP training we only need to sync gradients at the last micro
            # step. the official way to do this is with model.no_sync() context
            # manager, but I really dislike that this bloats the code and
            # forces us to repeat code looking at the source of that context
            # manager, it just toggles this variable
            # -----------------------------------------------------------------
            if self.config.train.backend.lower() == 'ddp':
                _ = (
                    self.model_engine.require_backward_grad_sync
                    if (is_last_micro_step and self.world_size > 1)
                    else None
                )
            fout = self._forward_step(x, y)
            # immediately async prefetch next batch while model is doing the
            # forward pass on the GPU
            x, y = self.get_batch('train')
            loss = fout['loss'] / self._gas
            dtf.append(fout['dt'])
            dtb_ = self._backward_step(loss, propagate_grads=is_last_micro_step)
            dtb.append(dtb_)
            dt.append(dtf + dtb)
        timers = {
            'iter': self.config.iter_num,
            'dt': np.array(dt),
            'dt_tot': np.sum(dt),
            'dt_avg': np.mean(dt),
            'dtf': np.array(dtf),
            'dtf_tot': np.sum(dtf),
            'dtf_avg': np.mean(dtf),
            'dtb': np.array(dtb),
            'dtb_tot': np.sum(dtb),
            'dtb_avg': np.mean(dtb),
        }
        metrics = {
            'iter': self.config.iter_num,
            'loss': loss,
            'lr': lr,
        }
        self.config.iter_num += 1
        return {
            'metrics': metrics,
            'timers': timers,
            'x': x,
            'y': y,
        }

    def save_ckpt(
        self,
        raw_model: Optional[torch.nn.Module | GPT] = None,
        add_to_wandb: bool = False,
    ):
        model = raw_model if raw_model is None else self.model
        # if raw_model is None:
        #     model = self.model
        # else:
        #     model = raw_model  # type:ignore
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        # assert issubclass(GPT,  torch.nn.Module)
        ckpt = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': asdict(self.config.model),
            'iter_num': self.config.iter_num,
            'best_val_loss': self.config.best_val_loss,
            'config': asdict(self.config),
        }
        # assert (
        #     isinstance(model, GPT)
        #     and issubclass(GPT, torch.nn.Module)
        # )
        # assert raw_model is not None
        ckptfile = Path(os.getcwd()).joinpath('ckpt.pt')
        modelfile = Path(os.getcwd()).joinpath('model.pth')
        log.info(f'Saving checkpoint to: {os.getcwd()}')
        log.info(f'Saving model to: {modelfile}')
        torch.save(model.state_dict(), modelfile.as_posix())
        torch.save(ckpt, ckptfile.as_posix())
        add_to_ckpts_file(Path(os.getcwd()))
        if add_to_wandb and wandb.run is not None:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(modelfile.as_posix())
            wandb.run.log_artifact(artifact)

    @timeitlogit(rank=RANK, verbose=(RANK != 0))
    def train(
        self,
        train_iters: Optional[int] = None,
    ):
        x, y = self.get_batch('train')
        t0 = time.perf_counter()
        running_mfu = -1.0
        output = {'x': x, 'y': y}
        t0 = time.perf_counter()
        losses = {}
        train_iters = (
            self.config.train.max_iters if train_iters is None else train_iters
        )
        for train_iter in trange(
            train_iters,
            disable=(self.rank != 0),
            total=train_iters,
        ):
            if self.config.iter_num == 0:
                start_time = os.environ.get('START_TIME', None)
                if start_time is not None:
                    startup_time = time.perf_counter() - float(start_time)
                    log.info(f'Startup time: {startup_time:.4f}')
                    if wandb is not None and wandb.run is not None:
                        wandb.run.log(
                            {'Timing/startup_time': startup_time}, commit=False
                        )
            if self.config.iter_num == 0 and self.config.train.eval_only:
                return
            if (
                self.config.iter_num % self.config.train.eval_interval == 0
                and self.rank == 0
            ):
                losses = self.estimate_loss()
                if self.config.iter_num > 0 and (
                    losses.get('val', np.inf) < self.config.best_val_loss
                    or self.config.train.always_save_checkpoint
                ):
                    self.save_ckpt(add_to_wandb=False)
            output = self.train_step(x=output['x'], y=output['y'])
            t1 = time.perf_counter()
            dt = t1 - t0
            tokens_per_sec = self.config.tokens_per_iter / dt
            samples_per_sec = self.config.samples_per_iter / dt
            t0 = t1
            output['timers'] |= {
                'dt_iter': dt,
                'tokens_per_sec': tokens_per_sec,
                'samples_per_sec': samples_per_sec,
            }
            # metrics = output['metrics']
            # metrics |= output['timers']
            lossf = output['metrics']['loss'].item() * self._gas
            output['metrics']['loss_tot'] = lossf
            _ = self.train_history.update(output['timers'])
            _ = self.train_history.update(output['metrics'])
            zero = torch.tensor(0.0)
            if self.config.iter_num % self.config.train.log_interval == 0 and (
                self.rank == 0
            ):
                if train_iter >= 5:
                    mfu = self.model.estimate_mfu(
                        (
                            self.config.model.batch_size
                            * self.config.optimizer.gas
                        ),
                        dt=dt,
                    )
                    running_mfu = (
                        mfu
                        if running_mfu == -1.0
                        else 0.9 * running_mfu + 0.1 * mfu
                    )
                pvars = {
                    'step': self.config.iter_num,
                    'loss': lossf,
                    'dt': dt * 1000,
                    'dtf': output['timers']['dtf_avg'] * 1000,
                    'dtb': output['timers']['dtb_avg'] * 1000,
                    'sps': samples_per_sec,
                    'mtps': tokens_per_sec / int(1e6),
                    'mfu': running_mfu * 100,
                    'train_loss': losses.get('train', zero).item(),
                    'val_loss': losses.get('val', zero).item(),
                }
                summary = summarize_dict(pvars)
                log.info(Text(summary))
                if wandb.run is not None:
                    losses |= {
                        'lossf': lossf,
                        'mfu': running_mfu * 100,
                        'iter': self.config.iter_num,
                    }
                    losses['lossf'] = lossf
                    losses['iter'] = self.config.iter_num
                    wbmetrics = (
                        {  # type:ignore
                            f'Training/{k}': (
                                wandb.Histogram(v.tolist())
                                if isinstance(v, np.ndarray)
                                else v
                            )
                            for k, v in output['metrics'].items()
                        }
                        | {
                            f'Timing/{k}': (
                                wandb.Histogram(v.tolist())
                                if isinstance(v, np.ndarray)
                                else v
                            )
                            for k, v in output['timers'].items()
                        }
                        | {f'Loss/{k}': v for k, v in losses.items()}
                    )
                    wandb.run.log(wbmetrics)
                    # wandb.run.log({
                    #     'losses': losses,
                    #     'metrics': output['metrics'],
                    #     'timers': output['timers'],
                    #     # 'training': metrics,
                    # })

    def evaluate(
        self,
        s: str,
        num_samples: int = 10,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_k: int = 200,
        display: Optional[bool] = True,
    ) -> dict[str, str]:
        # seed: Optional[int] = None,
        assert isinstance(self.model.module, GPT)
        assert issubclass(GPT, torch.nn.Module)
        self.model.eval()
        outputs = {}
        with torch.no_grad():
            start_ids = self.config.data.encode(s)
            x = torch.tensor(
                start_ids,
                dtype=torch.long,
                device=self.device,
            )[None, ...]
            for idx in range(num_samples):
                y = self.model.module.generate(
                    x, max_new_tokens, temperature=temperature, top_k=top_k
                )
                response = [
                    i
                    for i in self.config.data.decode(y[0].tolist()).split('\n')
                ]
                prompt = response[0]
                responses = [*response[1:]]
                ret0 = rf"[prompt]: '{prompt}'"
                ret1 = '> ' + '\n> '.join(responses)
                if display:
                    log.info(f'{ret0}')
                    log.info(f'{ret1}')
                outputs[f'{idx}'] = {
                    'raw': response,
                    'prompt': Text(ret0, style='string'),
                    'formatted': Text(ret1, style='blockquote'),
                }
                # log.info(f'[prompt]: "{s}"')
                # # responses = reponse.split('\n ')
                # log.info('> "' + '\n> '.join(response.split('\n ')) + '"')
                #
                # log.info('\n'.join)
                # log.info(f'> "{response}"')
                # log.info(100 * '-')
        return outputs
