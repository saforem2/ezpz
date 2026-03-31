"""Minimal feed-forward model used by :mod:`ezpz.examples.test` and other smoke tests.

Provides a simple variable-depth MLP that can be wrapped in FSDP / DDP
without any special handling, making it useful for validating distributed
training infrastructure.
"""

import torch


class SequentialLinearNet(torch.nn.Module):
    """Variable-depth MLP built from stacked ``Linear`` + ``ReLU`` layers.

    When *sizes* is ``None`` the network is a single linear projection from
    *input_dim* to *output_dim*.  Otherwise each entry in *sizes* adds a
    hidden layer, e.g. ``sizes=[256, 128]`` produces::

        Linear(input_dim, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, output_dim)

    Args:
        input_dim: Dimensionality of the input features.
        output_dim: Dimensionality of the output predictions.
        sizes: Optional list of hidden-layer widths.  ``None`` gives a
            single linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sizes: list[int] | None,
    ):
        super(SequentialLinearNet, self).__init__()
        nh = output_dim if sizes is None else sizes[0]
        layers = [torch.nn.Linear(input_dim, nh), torch.nn.ReLU()]
        if sizes is not None and len(sizes) > 1:
            for idx, size in enumerate(sizes[1:]):
                layers.extend([torch.nn.Linear(sizes[idx], size), torch.nn.ReLU()])
            layers.append(torch.nn.Linear(sizes[-1], output_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass through all layers.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Output tensor of shape ``(batch, output_dim)``.
        """
        return self.layers(x)


# class Network(torch.nn.Module):
#     def __init__(
#         self,
#         input_dim: int = 128,
#         output_dim: int = 128,
#         sizes: Optional[list[int]] = None,
#     ):
#         super(Network, self).__init__()
#         if sizes is None:
#             self.layers = torch.nn.Linear(input_dim, output_dim)
#         elif len(sizes) > 0:
#             layers = [torch.nn.Linear(input_dim, sizes[0])]
#             for idx, size in enumerate(sizes[1:]):
#                 layers.append(torch.nn.Linear(sizes[idx], size))
#             layers.append(torch.nn.Linear(sizes[-1], output_dim))
#             self.layers = torch.nn.Sequential(*layers)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.layers(x)
