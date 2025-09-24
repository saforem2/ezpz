# `ezpz.configs`

Configuration management for the ezpz package.

## Overview

The `ezpz.configs` module provides configuration management functionality,
including path definitions, scheduler detection, and training configuration classes.

## Key Components

::: ezpz.configs

## Usage Examples

### Path Management

```python
import ezpz.configs as configs

# Access important project paths
print(configs.PROJECT_DIR)      # Project root directory
print(configs.CONF_DIR)         # Configuration directory
print(configs.BIN_DIR)          # Binary/scripts directory

# Access configuration files
print(configs.DS_CONFIG_YAML)   # DeepSpeed config YAML
print(configs.DS_CONFIG_JSON)   # DeepSpeed config JSON
```

### Scheduler Detection

```python
import ezpz.configs as configs

# Detect the current job scheduler
scheduler = configs.get_scheduler()
print(scheduler)  # "PBS", "SLURM", or "UNKNOWN"
```

### Command Availability

```python
import ezpz.configs as configs

# Check if a command is available
if configs.command_exists("python"):
    print("Python is available")
else:
    print("Python is not available")
```

### Training Configuration

```python
import ezpz.configs as configs

# Create a training configuration
train_config = configs.TrainConfig(
    model_name_or_path="bert-base-uncased",
    output_dir="./outputs",
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=3,
)

print(train_config)  # Display configuration
```

### Logging Configuration

```python
import ezpz.configs as configs

# Get logging configuration
log_config = configs.get_logging_config()
print(log_config)  # Dictionary with logging configuration
```
