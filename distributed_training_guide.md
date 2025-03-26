# Distributed Training Guide with ezpz

This guide provides an overview of how to use ezpz for distributed training. We'll cover different backends (DDP, DeepSpeed, Horovod) and explain how to configure and launch distributed training jobs.

## Table of Contents

1. [Introduction to Distributed Training](#introduction-to-distributed-training)
2. [Setting Up ezpz](#setting-up-ezpz)
3. [Distributed Data Parallel (DDP)](#distributed-data-parallel-ddp)
4. [DeepSpeed](#deepspeed)
5. [Horovod](#horovod)
6. [Launching Distributed Training Jobs](#launching-distributed-training-jobs)
7. [Best Practices and Tips](#best-practices-and-tips)

## Introduction to Distributed Training

Distributed training allows you to leverage multiple GPUs or machines to train large models faster. ezpz simplifies the process of setting up and running distributed training jobs across different backends.

## Setting Up ezpz

To use ezpz for distributed training, first import the necessary functions:

```python
from ezpz import setup, get_rank, get_world_size, get_local_rank, get_torch_device
```

## Distributed Data Parallel (DDP)

DDP is the recommended method for multi-GPU training in PyTorch. Here's how to set it up with ezpz:

1. Initialize the distributed environment:

```python
rank = setup(framework='pytorch', backend='DDP')
```

2. Move your model to the correct device and wrap it with DDP:

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

model = YourModel()
device = get_torch_device()
local_rank = get_local_rank()
device_id = f'{device}:{local_rank}'
model.to(device_id)
model = DDP(model, device_ids=[device_id])
```

3. Use DistributedSampler for your DataLoader:

```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
)
```

## DeepSpeed

DeepSpeed is an optimization library that makes distributed training even easier. To use it with ezpz:

1. Initialize the distributed environment:

```python
rank = setup(framework='pytorch', backend='deepspeed')
```

2. Create a DeepSpeed configuration file (e.g., `ds_config.json`):

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [0.9, 0.999],
      "eps": 1e-8
    }
  },
  "fp16": {
    "enabled": true
  }
}
```

3. Initialize your model and optimizer with DeepSpeed:

```python
import deepspeed

model = YourModel()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)
```

4. Use the `model_engine` for training:

```python
for data, target in train_loader:
    outputs = model_engine(data)
    loss = criterion(outputs, target)
    model_engine.backward(loss)
    model_engine.step()
```

## Horovod

Horovod is another popular distributed training framework. Here's how to use it with ezpz:

1. Initialize the distributed environment:

```python
rank = setup(framework='tensorflow', backend='horovod')
```

2. Initialize Horovod and wrap your optimizer:

```python
import horovod.tensorflow as hvd

hvd.init()

optimizer = tf.keras.optimizers.Adam(lr=0.001 * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)
```

3. Scale your learning rate and use Horovod callbacks:

```python
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(
        warmup_epochs=5, initial_lr=0.001 * hvd.size(), verbose=1),
]

model.fit(dataset,
          steps_per_epoch=500 // hvd.size(),
          callbacks=callbacks,
          epochs=24,
          verbose=1 if hvd.rank() == 0 else 0)
```

## Launching Distributed Training Jobs

The exact command to launch distributed training jobs depends on your cluster setup. Here are some examples:

- For DDP on a single machine with 4 GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=4 your_script.py
```

- For DeepSpeed on multiple nodes:

```bash
deepspeed --num_gpus=8 --num_nodes=2 your_script.py
```

- For Horovod on multiple nodes:

```bash
horovodrun -np 16 -H server1:8,server2:8 python your_script.py
```

## Best Practices and Tips

1. Use `get_rank()` to ensure only one process performs certain operations (e.g., logging, saving checkpoints):

```python
if get_rank() == 0:
    # Perform operation only on the main process
    save_checkpoint(model)
```

2. Scale your learning rate based on the world size:

```python
lr = base_lr * get_world_size()
```

3. Use `timeitlogit` decorator for easy performance logging:

```python
from ezpz import timeitlogit

@timeitlogit(verbose=True)
def train_epoch():
    # Your training code here
    pass
```

4. Utilize ezpz's built-in functions for device management:

```python
device = get_torch_device()
local_rank = get_local_rank()
```

By following this guide, you should be able to set up and run distributed training jobs using ezpz with various backends. Remember to adjust your code and configuration based on your specific use case and cluster setup.