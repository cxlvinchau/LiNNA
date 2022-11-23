import abc
from typing import List

from linna.network import Network


class BasisFinder(abc.ABC):
    """Basis finder base class"""

    def __init__(self, network: Network, **parameters):
        self.network = network
        self.parameters = parameters

    @abc.abstractmethod
    def find_basis(self, layer_idx: int) -> List[int]:
        """
        Finds the basis for the given layer

        Parameters
        ----------
        layer_idx: int
            Layer

        Returns
        -------
        List[int]
            Basis for layer

        """
        pass
