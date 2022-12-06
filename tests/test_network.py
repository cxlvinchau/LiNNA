import unittest

import torch
from torch import nn

from linna.network import Network
from tests.toy_network import create_toy_network


class TestNetwork(unittest.TestCase):

    def setUp(self) -> None:
        # Wrap PyTorch model
        self.network = create_toy_network()
        self.sequential = self.network.torch_model

    def test_init_network(self):
        assert len(self.network.layers) == 3, "Wrong numbers of layers"
        for layer in self.network.layers:
            assert isinstance(self.sequential[layer.layer_idx], nn.Linear)

    def test_restore_delete_neuron(self):
        # Delete neuron
        self.network.delete_neuron(layer_idx=0, neuron=1)

        # Assert that weight dimensions are correct
        assert self.network.layers[0].get_weight().size() == torch.Size((2, 2)), "Weight not correct"
        assert self.network.layers[1].get_weight().size() == torch.Size((3, 2)), "Successor weight not correct"

        # Assert that bias dimensions are correct
        assert self.network.layers[0].get_bias().size() == torch.Size((2,)), "Bias not correct"

        # Assert neuron has been removed from active neurons
        assert 1 not in self.network.layers[0].active_neurons, "Neuron not removed"

    def test_forward(self):
        assert torch.all(self.network.forward(torch.Tensor([[1, 0]])) == torch.Tensor([[16, 16, 16]]))

    def test_replace(self):
        self.network.delete_neuron(layer_idx=0, neuron=1)
        self.network.set_basis(layer_idx=0, basis=[0, 2])
        self.network.readjust_weights(layer_idx=0, neuron=1, coef=torch.Tensor([1, 2]))
        assert torch.all(torch.isclose(self.network.layers[1].get_weight(), torch.tensor([[1., 0.], [1., 2.], [0., 1.]])))

    def test_get_io_matrix(self):
        io_matrix = self.network.get_io_matrix(1, loader=[[torch.tensor([[1., 0.], [2., -1.]]), [0, 1]]], size=2)
        assert torch.all(torch.isclose(torch.tensor(io_matrix), torch.tensor([[3., 4., 7.], [2., 4., 7.]])))


if __name__ == '__main__':
    unittest.main()
