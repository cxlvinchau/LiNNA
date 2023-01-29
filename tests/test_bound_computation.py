import pytest
import torch

from linna.abstraction import Abstraction
from linna.verification.error_computation import compute_bounds
from tests.toy_network import create_toy_network


@pytest.fixture
def network():
    return create_toy_network()


@pytest.fixture
def loader():
    return [[torch.tensor([[1., 0.], [2., -1.], [3., 5.], [1., 1.]]), [0, 1, 0, 0]]]


class TestBoundComputation:
    def test_bound_computation(self, network, loader):
        x = loader[0][0][0].unsqueeze(0)
        lb, ub = compute_bounds(network=network, x=x, epsilon=0.05, layer_idx=0)
        print(lb, ub)

