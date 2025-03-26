# Logging and Monitoring in ezpz

This guide explains how to use ezpz's logging and monitoring features, including customizing logging output, integrating with Weights & Biases (wandb), and tracking training progress effectively.

## Table of Contents

1. [Setting Up Logging](#setting-up-logging)
2. [Customizing Logging Output](#customizing-logging-output)
3. [Integrating with Weights & Biases](#integrating-with-weights--biases)
4. [Tracking Training Progress](#tracking-training-progress)

## Setting Up Logging

ezpz uses Python's built-in `logging` module along with custom handlers for enhanced functionality. To set up logging in your project:

```python
import logging
from ezpz.log import get_logger

# Create a logger
log = get_logger(__name__, level="INFO")
```

The `get_logger` function sets up a logger with ezpz's custom configurations, including colored output and rich formatting.

## Customizing Logging Output

ezpz provides several ways to customize your logging output:

### Log Levels

You can set different log levels to control the verbosity of your logs:

```python
log.setLevel(logging.DEBUG)  # Set to DEBUG, INFO, WARNING, ERROR, or CRITICAL
```

### Colored Logs

ezpz uses the `rich` library to provide colored and formatted logs. You can customize the appearance of your logs using ezpz's predefined styles:

```python
from ezpz.log import STYLES

# Example of using a custom style
log.info("This is an important message", style=STYLES["important"])
```

### Custom Formatting

To create custom log formats, you can use ezpz's `CustomLogging` class:

```python
from ezpz.log import CustomLogging

custom_logger = CustomLogging(name="custom_logger", level="INFO")
custom_logger.info("This is a custom formatted log message")
```

## Integrating with Weights & Biases

ezpz provides built-in support for logging to Weights & Biases (wandb). To use this feature:

1. Install wandb: `pip install wandb`
2. Initialize wandb in your script:

```python
import wandb

wandb.init(project="your_project_name")
```

3. Log metrics and other data to wandb:

```python
wandb.log({"loss": loss_value, "accuracy": acc_value})
```

In the `TrainerLLM` class, ezpz automatically logs various metrics to wandb if it's initialized:

```python
if wandb.run is not None:
    wbmetrics = {
        f'Training/{k}': v for k, v in output['metrics'].items()
    } | {
        f'Timing/{k}': v for k, v in output['timers'].items()
    } | {
        f'Loss/{k}': v for k, v in losses.items()
    }
    wandb.run.log(wbmetrics)
```

## Tracking Training Progress

ezpz provides various ways to track and visualize training progress:

### Console Output

The `TrainerLLM` class prints formatted progress information to the console:

```python
summary = summarize_dict(pvars)
log.info(Text(summary))
```

This includes metrics such as loss, learning rate, tokens per second, and model utilization.

### History Tracking

ezpz uses a `History` class to keep track of training metrics over time:

```python
self.train_history = ezpz.History()
_ = self.train_history.update(output['timers'])
_ = self.train_history.update(output['metrics'])
```

You can access and analyze this history data to gain insights into your training process.

### Checkpointing

ezpz automatically saves checkpoints during training, which can be used to resume training or evaluate models:

```python
self.save_ckpt(add_to_wandb=False)
```

By leveraging these logging and monitoring features, you can effectively track your model's training progress, debug issues, and share results with your team or the broader community.