import unittest

import torch
from torch import nn

from linna.network import Network


class TestNetwork(unittest.TestCase):

    def setUp(self) -> None:
        self.sequential = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 3)
        )

        # Set weights
        with torch.no_grad():
            self.sequential[0].weight = nn.Parameter(torch.Tensor([[1, 2], [1, 1], [3, 3]]))
            self.sequential[0].bias = nn.Parameter(torch.Tensor([1, 2, 3]))
            self.sequential[2].weight = nn.Parameter(torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            self.sequential[2].bias = nn.Parameter(torch.ones(3))
            self.sequential[4].weight = nn.Parameter(torch.ones((3, 3)))
            self.sequential[4].bias = nn.Parameter(2 * torch.ones(3))

        # Wrap PyTorch model
        self.network = Network(torch_model=self.sequential)

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
        self.network.readjust_weights(layer_idx=0, neuron=0, coef=torch.Tensor([1, 1]))
        print(self.network.layers[1].get_weight())


if __name__ == '__main__':
    unittest.main()
