import torch
from torch import nn

from linna.network import Network


def create_toy_network():
    sequential = nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 3),
        nn.ReLU(),
        nn.Linear(3, 3)
    )

    # Set weights
    with torch.no_grad():
        sequential[0].weight = nn.Parameter(torch.Tensor([[1, 2], [1, 1], [3, 3]]))
        sequential[0].bias = nn.Parameter(torch.Tensor([1, 2, 3]))
        sequential[2].weight = nn.Parameter(torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        sequential[2].bias = nn.Parameter(torch.ones(3))
        sequential[4].weight = nn.Parameter(torch.ones((3, 3)))
        sequential[4].bias = nn.Parameter(2 * torch.ones(3))

    # Wrap PyTorch model
    return Network(torch_model=sequential)
