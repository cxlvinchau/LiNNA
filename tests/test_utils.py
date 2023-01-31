import pytest
from torch import nn
import torch

from linna.utils import is_real_cex
from tests.toy_network import create_toy_network

@pytest.fixture
def identity_network():
    sequential = nn.Sequential(
        nn.Linear(3, 3),
    )

    # Set weights
    with torch.no_grad():
        sequential[0].weight = nn.Parameter(torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        sequential[0].bias = nn.Parameter(torch.zeros(3))

    return sequential


class TestUtils:
    def test_is_real_cex(self, identity_network):
        for i in range(3):
            x = torch.zeros((1, 3))
            x[0][i] = 1
            assert not is_real_cex(network=identity_network, cex=x, target_cls=i)

        for i in range(3):
            x = torch.ones((1, 3))
            x[0][i] = 0
            assert is_real_cex(network=identity_network, cex=x, target_cls=i)

        for i in range(3):
            assert is_real_cex(network=identity_network, cex=torch.ones((1, 3)), target_cls=i)

        for i in range(3):
            x = torch.ones((1, 3))
            x[0][i] = 2 - i
            assert is_real_cex(network=identity_network, cex=torch.ones((1, 3)), target_cls=i)
