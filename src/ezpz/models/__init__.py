"""Model zoo for ezpz examples and benchmarks.

Submodules expose Llama-style transformers, vision transformers, attention
primitives, and a minimal MLP used by smoke tests.
"""

import ezpz
from typing import Optional, Sequence
from ezpz.models import (
    attention,
    llama,
    llama3,
    minimal,
    vit,
)
# from ezpz.models.attention import (
#     FlexAttention,
#     ScaledDotProductAttention,,
#         build_attention,
#     init_attention_mask,
# )

logger = ezpz.get_logger(__name__)

__all__ = [
    "attention",
    "llama",
    "llama3",
    "minimal",
    "vit"
]


def summarize_model(
    model: "torch.nn.Module",
    verbose: bool = False,
    depth: int = 1,
    input_size: Optional[Sequence[int]] = None,
) -> object | None:
    """Print a ``torchinfo`` model summary if available.

    Args:
        model: The ``nn.Module`` to summarise.
        verbose: Passed through to ``torchinfo.summary``.
        depth: Depth of nested module display.
        input_size: Optional input size for shape inference.

    Returns:
        The ``torchinfo.ModelStatistics`` object, or ``None`` if
        ``torchinfo`` is not installed.
    """
    try:
        import torchinfo
        from torchinfo import summary

        summary_str = summary(
            model,
            input_size=input_size,
            depth=depth,
            verbose=verbose,
        )
        # logger.info(f'\n{summary_str}')
        return summary_str

    except (ImportError, ModuleNotFoundError):
        logger.warning(
            'torchinfo not installed, unable to print model summary!'
        )
