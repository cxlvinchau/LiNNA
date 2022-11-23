import abc

import torch
import warnings
import logging
import numpy as np

from linna.basis_finder import BasisFinder
from linna.network import Network


class ReductionBaseClass(abc.ABC):
    """Reduction base class"""

    def __init__(self, network: Network, loader, size: int = 1000):
        self.model = network
        self.loader = loader
        self.size = size
        self.io_matrix = dict()
        for layer in range(len(network.layers)):
            self.io_matrix[layer] = network.get_io_matrix(loader, layer, size)

    @abc.abstractmethod
    def process_layer(self, layer_idx: int, **parameters):
        """
        Processes the given layer with the specified parameters

        Parameters
        ----------
        layer_idx: int
            Layer
        parameters: Dict[str, Any]
            Parameters for the algorithm

        """
        pass


class L1Reduction(ReductionBaseClass):
    """Reduction of neural network by finding a linear combination via linear programming

    This class implements the linear programming reduction that replaces a given neuron by a linear combination of
    other neurons. This combination is minimal w.r.t. L1 distance.
    """

    def __init__(self, network: Network, loader, size=1000, seed=0):
        super().__init__(network, loader, size)
        self.removed_neurons = dict()
        self.rng = np.random.default_rng(seed)

    def process_neuron(self, layer_idx: int, neuron: int):
        """
        Processes a single neuron of a given layer

        Parameters
        ----------
        layer_idx: int
            Layer
        neuron: int
            Neuron

        """



    def process_layer(self, layer_idx, **parameters):

        # Removed and active neurons
        removed_neurons = self.removed_neurons.setdefault(layer_idx, [])
        active_neurons = [i for i in range(self.io_matrix[layer_idx].shape[1]) if i not in removed_neurons and i != neuron]

        io_matrix = self.io_matrix[layer_idx][:, active_neurons]
        idx_to_neuron = [i for i in self.model.layers[layer_idx].active_neurons if i != neuron]
        solution = None
        with gb.Env(empty=True) as env:
            env.setParam("LogToConsole", 0)
            env.start()
            with gb.Model(env=env) as grb_model:
                io_vars = grb_model.addMVar(io_matrix.shape[1], lb=float("-inf"), name="io_vars")
                pos_slack = grb_model.addMVar(io_matrix.shape[0], lb=0, name="pos_slack")
                neg_slack = grb_model.addMVar(io_matrix.shape[0], lb=0, name="neg_slack")
                target = self.io_matrix[layer_idx][:, neuron]
                # Add linear combination constraint
                print(io_matrix.shape)
                print(io_vars)
                lin_expr = (io_matrix @ io_vars) - (identity(io_matrix.shape[0]) @ pos_slack) + (
                        identity(io_matrix.shape[0]) @ neg_slack)
                grb_model.addConstr(lin_expr == target)
                # Specify the objective, expressed by slack variables (corresponds to L1 norm distance)
                grb_model.setObjective(
                    np.ones(io_matrix.shape[0]) @ pos_slack + np.ones(io_matrix.shape[0]) @ neg_slack, GRB.MINIMIZE)
                grb_model.optimize()
                solution = io_vars.x

        # Current input weights of neuron
        input_weight = self.model.layers[layer_idx+1].get_input_weight(neuron).detach().clone()
        change_matrix = torch.diag(input_weight)

        self.removed_neurons[layer_idx].append(neuron)
        self.model.layers[layer_idx].delete_output(neuron, succ_layer=self.model.layers[layer_idx + 1])

        # Re-adjust weights
        change_matrix = torch.matmul(change_matrix, torch.tensor(solution).repeat(change_matrix.shape[0], 1).float())
        print(change_matrix.shape)
        print(self.model.layers[layer_idx+1].get_weight().shape)
        self.model.layers[layer_idx+1].set_weight(self.model.layers[layer_idx+1].get_weight() + change_matrix)


