import pytest
import torch

from linna.abstraction import Abstraction
from tests.toy_network import create_toy_network


@pytest.fixture
def network():
    return create_toy_network()


@pytest.fixture
def loader():
    return [[torch.tensor([[1., 0.], [2., -1.], [3., 5.], [1., 1.]]), [0, 1, 0, 0]]]


@pytest.mark.parametrize("basis_finder", ["variance", "greedy"])
@pytest.mark.parametrize("coef_finder", ["l1", "l2"])
class TestAbstraction:

    def test_abstract(self, basis_finder, coef_finder, network, loader):
        abstraction = Abstraction(
            network=network,
            basis_finder=basis_finder,
            coef_finder=coef_finder,
            loader=loader,
            size=4
        )
        abstraction.determine_basis(layer_idx=0, basis_size=2)
        abstraction.abstract(layer_idx=0)
        assert abstraction.network.layers[0].get_weight().size() == torch.Size([2, 2])
        assert abstraction.network.layers[1].get_weight().size() == torch.Size([3, 2])
        assert abstraction.network.layers[0].get_bias().size() == torch.Size([2])
        assert abstraction.network.layers[1].get_bias().size() == torch.Size([3])
