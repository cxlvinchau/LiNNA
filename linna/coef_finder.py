import abc
from typing import Dict, Any

import torch

from linna.network import Network
import gurobipy as gb
from gurobipy import GRB
from scipy.sparse import identity, hstack, coo_matrix
import numpy as np

from scipy.optimize import linprog


class _CoefFinder(abc.ABC):
    """Reduction base class"""

    def __init__(self, network: Network = None, io_dict: Dict[int, np.ndarray] = None, params: Dict[str, Any] = None):
        self.network = network
        self.io_dict = io_dict
        self.params = params

    def find_coefficients_layer(self, layer_idx: int) -> Dict[int, torch.Tensor]:
        result = dict()
        for neuron in self.network.layers[layer_idx].neurons:
            if neuron not in self.network.layers[layer_idx].basis:
                result[neuron] = self.find_coefficients(layer_idx=layer_idx, neuron=neuron)
        return result

    @abc.abstractmethod
    def find_coefficients(self, layer_idx: int, neuron: int) -> torch.Tensor:
        """
        Processes the given layer with the specified parameters

        Parameters
        ----------
        layer_idx: int
            Layer
        neuron: int
            Neuron

        Returns
        -------
        torch.Tensor
            Coefficients

        """
        pass


class L1CoefFinder(_CoefFinder):

    def _find_coefficients_gurobi(self, layer_idx: int, neuron: int):
        io_matrix = self.io_dict[layer_idx][:, self.network.layers[layer_idx].basis]
        with gb.Env(empty=True) as env:
            env.setParam("LogToConsole", 0)
            env.start()
            with gb.Model(env=env) as grb_model:
                io_vars = grb_model.addMVar(io_matrix.shape[1], lb=float("-inf"), name="io_vars")
                pos_slack = grb_model.addMVar(io_matrix.shape[0], lb=0, name="pos_slack")
                neg_slack = grb_model.addMVar(io_matrix.shape[0], lb=0, name="neg_slack")
                target = self.io_dict[layer_idx][:, neuron]
                # Add linear combination constraint
                lin_expr = (io_matrix @ io_vars) - (identity(io_matrix.shape[0]) @ pos_slack) + (
                        identity(io_matrix.shape[0]) @ neg_slack)
                grb_model.addConstr(lin_expr == target)
                # Specify the objective, expressed by slack variables (corresponds to L1 norm distance)
                grb_model.setObjective(
                    np.ones(io_matrix.shape[0]) @ pos_slack + np.ones(io_matrix.shape[0]) @ neg_slack, GRB.MINIMIZE)
                grb_model.optimize()
                return torch.Tensor(io_vars.x)

    def _find_coefficients_scipy(self, layer_idx: int, neuron: int):
        io_matrix = coo_matrix(self.io_dict[layer_idx][:, self.network.layers[layer_idx].basis])
        identity_mat = identity(io_matrix.shape[0])
        identity_mat_neg = -1 * identity(io_matrix.shape[0])
        # Constraint matrix
        A_eq = hstack([io_matrix, identity_mat, identity_mat_neg])
        b_eq = self.io_dict[layer_idx][:, neuron]
        # Only slack variables are bounded
        bounds = [(None, None) for _ in range(io_matrix.shape[1])] + [(0, None) for _ in range(2 * io_matrix.shape[0])]
        # Objective function
        c = np.concatenate((np.zeros(io_matrix.shape[1]), np.ones(2 * io_matrix.shape[0])))
        result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        return torch.tensor(result.x[:io_matrix.shape[1]]).float()

    def find_coefficients(self, layer_idx: int, neuron: int) -> torch.Tensor:
        if "solver" in self.params:
            if self.params["solver"] == "gurobi":
                return self._find_coefficients_gurobi(layer_idx=layer_idx, neuron=neuron)
            if self.params["solver"] == "scipy":
                return self._find_coefficients_scipy(layer_idx=layer_idx, neuron=neuron)
            raise ValueError(f"Unsupported solver {self.params['solver']}")

        return self._find_coefficients_gurobi(layer_idx=layer_idx, neuron=neuron)


class L2CoefFinder(_CoefFinder):

    def __init__(self, network: Network = None, io_dict: Dict[int, np.ndarray] = None):
        super().__init__(network=network, io_dict=io_dict)
        self._inverse = None
        self._basis = None

    def find_coefficients(self, layer_idx: int, neuron: int, **parameters) -> torch.Tensor:
        io_matrix = self.io_dict[layer_idx]
        basis = self.network.layers[layer_idx].basis
        # Cache matrix
        if basis != self._basis:
            A = io_matrix[:, basis]
            X = np.linalg.inv(np.matmul(A.T, A))
            self._basis = basis
            self._inverse = X
        A = io_matrix[:, basis]
        X = np.matmul(self._inverse, np.matmul(A.T, io_matrix[:, neuron]))
        return torch.Tensor(X)

    def find_coefficients_layer(self, layer_idx: int, **parameters) -> Dict[int, torch.Tensor]:
        io_matrix = self.io_dict[layer_idx]
        basis = self.network.layers[layer_idx].basis
        non_basic = [neuron for neuron in self.network.layers[layer_idx].neurons if neuron not in basis]
        A = io_matrix[:, basis]
        X = torch.Tensor(np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, io_matrix[:, non_basic])))
        return {neuron: X[:, idx] for idx, neuron in enumerate(non_basic)}


class InfinityCoefFinder(_CoefFinder):

    def find_coefficients(self, layer_idx: int, neuron: int, **parameters) -> torch.Tensor:
        pass
