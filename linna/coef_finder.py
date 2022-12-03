import abc

import torch

from linna.network import Network
import gurobipy as gb
from gurobipy import GRB
from scipy.sparse import identity
import numpy as np


class CoefFinder(abc.ABC):
    """Reduction base class"""

    def __init__(self, network: Network, loader, size: int = 1000):
        self.network = network
        self.loader = loader
        self.size = size
        self.io_matrix = dict()
        for layer in range(len(network.layers)):
            self.io_matrix[layer] = network.get_io_matrix(loader, layer, size)

    @abc.abstractmethod
    def find_coefficients(self, layer_idx: int, neuron: int, **parameters) -> torch.Tensor:
        """
        Processes the given layer with the specified parameters

        Parameters
        ----------
        layer_idx: int
            Layer
        neuron: int
            Neuron
        parameters: Dict[str, Any]
            Parameters for the algorithm

        Returns
        -------
        torch.Tensor
            Coefficients

        """
        pass


class L1CoefFinder(CoefFinder):

    def find_coefficients(self, layer_idx: int, neuron: int, **parameters) -> torch.Tensor:
        if neuron not in self.network.layers[layer_idx].active_neurons:
            raise ValueError("Neuron is not active")
        if neuron in self.network.layers[layer_idx].basis:
            raise ValueError("Neuron is part of basis")

        removed = self.network.layers[layer_idx].removed_neurons
        active = [i for i in range(self.io_matrix[layer_idx].shape[1]) if i not in removed and i != neuron]

        io_matrix = self.io_matrix[layer_idx][:, active]
        with gb.Env(empty=True) as env:
            env.setParam("LogToConsole", 0)
            env.start()
            with gb.Model(env=env) as grb_model:
                io_vars = grb_model.addMVar(io_matrix.shape[1], lb=float("-inf"), name="io_vars")
                pos_slack = grb_model.addMVar(io_matrix.shape[0], lb=0, name="pos_slack")
                neg_slack = grb_model.addMVar(io_matrix.shape[0], lb=0, name="neg_slack")
                target = self.io_matrix[layer_idx][:, neuron]
                # Add linear combination constraint
                lin_expr = (io_matrix @ io_vars) - (identity(io_matrix.shape[0]) @ pos_slack) + (
                        identity(io_matrix.shape[0]) @ neg_slack)
                grb_model.addConstr(lin_expr == target)
                # Specify the objective, expressed by slack variables (corresponds to L1 norm distance)
                grb_model.setObjective(
                    np.ones(io_matrix.shape[0]) @ pos_slack + np.ones(io_matrix.shape[0]) @ neg_slack, GRB.MINIMIZE)
                grb_model.optimize()
                return torch.Tensor(io_vars.x)


class L2CoefFinder(CoefFinder):

    def find_coefficients(self, layer_idx: int, neuron: int, **parameters) -> torch.Tensor:
        io = self.io_matrix[layer_idx]
        basis = self.network.layers[layer_idx].basis
        A = io[:, basis]
        X = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, io))
        return basis, X

