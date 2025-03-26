# Configuration Guide

This guide explains how to configure ezpz for different training scenarios using the `TrainConfig` class. You'll learn about the available properties and how to customize them for your specific needs.

## TrainConfig Class

The `TrainConfig` class is the main configuration object used in ezpz. It contains various properties that control the behavior of your training process.

### Properties

- `gas` (int): Global Accumulation Steps. Default is 1.
- `framework` (str): The deep learning framework to use. Options are 'tensorflow' or 'pytorch'. Default is 'pytorch'.
- `backend` (str): The distributed training backend. Options depend on the framework. Default is 'DDP' for PyTorch.
- `use_wandb` (bool): Whether to use Weights & Biases for logging. Default is False.
- `seed` (Optional[int]): Random seed for reproducibility. Default is None.
- `port` (Optional[str]): Port for distributed training. Default is None.
- `dtype` (Optional[Any]): Data type for training. Default is None.
- `load_from` (Optional[str]): Path to load a pre-trained model. Default is None.
- `save_to` (Optional[str]): Path to save the trained model. Default is None.
- `ds_config_path` (Optional[str]): Path to DeepSpeed configuration file. Default is None.
- `wandb_project_name` (Optional[str]): Weights & Biases project name. Default is None.
- `ngpus` (Optional[int]): Number of GPUs to use. Default is None.

## Common Configurations

Here are some examples of common configurations for different scenarios:

### Basic PyTorch Configuration

```python
from ezpz.configs import TrainConfig

config = TrainConfig(
    framework='pytorch',
    backend='DDP',
    seed=42,
    ngpus=4
)
```

This configuration sets up a PyTorch training job using DistributedDataParallel (DDP) on 4 GPUs with a fixed random seed.

### TensorFlow with Horovod

```python
config = TrainConfig(
    framework='tensorflow',
    backend='horovod',
    use_wandb=True,
    wandb_project_name='my_tf_project'
)
```

This configuration sets up a TensorFlow training job using Horovod for distributed training and enables Weights & Biases logging.

### DeepSpeed Configuration

```python
config = TrainConfig(
    framework='pytorch',
    backend='deepspeed',
    ds_config_path='/path/to/ds_config.json',
    gas=4,
    dtype='float16'
)
```

This configuration sets up a PyTorch training job using DeepSpeed, with a custom DeepSpeed configuration file, gradient accumulation, and mixed precision training.

## Customizing TrainConfig

You can customize the `TrainConfig` object by passing arguments when initializing it or by modifying its properties after creation:

```python
config = TrainConfig(framework='pytorch', backend='DDP')
config.use_wandb = True
config.wandb_project_name = 'my_custom_project'
config.seed = 12345
```

## Loading and Saving Configurations

TrainConfig provides methods to load and save configurations to files:

```python
# Save configuration to a file
config.to_file('my_config.json')

# Load configuration from a file
new_config = TrainConfig()
new_config.from_file('my_config.json')
```

## Printing Configurations

You can easily print your configuration for debugging or logging purposes:

```python
from ezpz.configs import print_config

print_config(config.to_dict())
```

This will output a formatted JSON representation of your configuration.

## Framework and Backend Compatibility

When setting up your configuration, keep in mind the following compatibility rules:

- For TensorFlow, the only supported backend is 'horovod'.
- For PyTorch, supported backends are 'DDP', 'deepspeed', and 'horovod'.

The `TrainConfig` class will automatically validate these combinations when you create or modify the configuration.

By using the `TrainConfig` class and following this guide, you can easily set up and customize your ezpz training configurations for various scenarios and frameworks.