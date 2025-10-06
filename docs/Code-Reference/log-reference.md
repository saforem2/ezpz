# `ezpz.log`

Logging utilities built on top of Rich.

::: ezpz.log

## Usage Examples

### Create a Rank-Aware Logger

```python
import ezpz.log as ezlog

logger = ezlog.get_logger("train")
logger.info("hello from rank 0")
```

### Print Available Styles

```python
from ezpz.log import print_styles

print_styles()
```
