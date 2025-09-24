# Launch Module Tests

## Overview

The launch module tests (`test_launch.py`) verify the command launching functionality, including scheduler detection and bash command execution.

## Test Cases

### test_command_exists

Tests the command existence checking function.

```python
def test_command_exists(self):
    """Test command_exists function."""
    # Test with a command that should exist
    assert launch.command_exists("python") is True
    
    # Test with a command that should not exist
    assert launch.command_exists("nonexistent_command_xyz") is False
```

### test_get_scheduler

Verifies the scheduler detection functionality.

```python
def test_get_scheduler(self):
    """Test get_scheduler function."""
    scheduler = launch.get_scheduler()
    assert isinstance(scheduler, str)
    # Should be one of the known schedulers or "UNKNOWN"
    assert scheduler in ["PBS", "SLURM", "UNKNOWN"]
```

### test_run_bash_command

Tests the bash command execution function.

```python
def test_run_bash_command(self):
    """Test run_bash_command function."""
    # Test a simple command
    result = launch.run_bash_command("echo 'test'")
    assert result is not None
```

### test_get_scheduler_from_pbs

Verifies scheduler detection from PBS environment variables.

```python
def test_get_scheduler_from_pbs(self, mock_pbs_env):
    """Test get_scheduler function with PBS environment."""
    # Set PBS environment variables
    os.environ["PBS_JOBID"] = "test.job"
    scheduler = launch.get_scheduler()
    assert scheduler == "PBS"
```

### test_get_scheduler_from_slurm

Tests scheduler detection from SLURM environment variables.

```python
def test_get_scheduler_from_slurm(self):
    """Test get_scheduler function with SLURM environment."""
    # Save original environment
    original_env = os.environ.copy()
    
    # Set SLURM environment variables
    os.environ["SLURM_JOB_ID"] = "test.job"
    scheduler = launch.get_scheduler()
    assert scheduler == "SLURM"
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
```

## Running Tests

```bash
python -m pytest tests/test_launch.py
```