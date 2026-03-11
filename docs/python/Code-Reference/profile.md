# `ezpz.profile`

- See [ezpz/`profile.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/profile.py)

Profiling utilities for distributed PyTorch training. Supports both
`torch.profiler` and `pyinstrument` backends.

## `get_profiling_context()`

Returns a context manager for profiling. The `profiler_type` parameter selects
the backend:

??? example "Torch profiler backend"

    With `profiler_type="torch"`, the returned context manager wraps
    `torch.profiler.profile`. Call `.step()` on the **torch profiler object**
    (not the context manager) to advance the profiling schedule:

    ```python
    from ezpz.profile import get_profiling_context

    with get_profiling_context(
        profiler_type="torch",
        wait=1,
        warmup=1,
        active=3,
        repeat=1,
        rank_zero_only=True,
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for step, batch in enumerate(dataloader):
            loss = model(batch)
            loss.backward()
            optimizer.step()
            if prof is not None:
                prof.step()
    ```

??? example "PyInstrument backend"

    With `profiler_type="pyinstrument"`, the context manager wraps
    `PyInstrumentProfiler`. There is no `.step()` method — just enter and
    exit the context:

    ```python
    from ezpz.profile import get_profiling_context

    with get_profiling_context(
        profiler_type="pyinstrument",
        rank_zero_only=True,
    ):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
    ```

## `get_torch_profiler()`

Lower-level wrapper around `torch.profiler.profile` with schedule and
trace-ready callback support:

```python
from ezpz.profile import get_torch_profiler

profiler = get_torch_profiler(
    rank=0,
    rank_zero_only=True,
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
)
```

The profiler writes traces that can be viewed in TensorBoard or Chrome's
`chrome://tracing` viewer.

## `PyInstrumentProfiler`

A context manager wrapping `pyinstrument.Profiler` for statistical profiling
with minimal overhead:

```python
from ezpz.profile import PyInstrumentProfiler

with PyInstrumentProfiler(rank=0, rank_zero_only=True):
    train_loop()

# Results written as HTML + text to ./ezpz_pyinstrument_profiles/
```

Output is always written to `ezpz_pyinstrument_profiles/` in the current
working directory (with timestamped filenames).

## `PYINSTRUMENT_PROFILER` Environment Variable

When using `get_profiling_context()` with `strict=True`, profiling is only
activated if the `PYINSTRUMENT_PROFILER` environment variable is set to `"1"`:

```bash
PYINSTRUMENT_PROFILER=1 ezpz launch -- python train.py
```

This allows you to add profiling instrumentation to your code without it
activating in every run.

::: ezpz.profile
