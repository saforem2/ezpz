"""
ezpz/models/minimal.py
"""
import torch


class SequentialLinearNet(torch.nn.Module):
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
                layers.extend(
                    [torch.nn.Linear(sizes[idx], size), torch.nn.ReLU()]
                )
            layers.append(torch.nn.Linear(sizes[-1], output_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
