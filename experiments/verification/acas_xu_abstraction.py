import sys, os
import numpy as np
from torchvision import datasets, transforms

from experiments.playground.bounds import lp_lower_bound, lp_upper_bound
from experiments.playground.marabou_network import linna_to_marabou_network
from experiments.playground.marabou_utils import linna_to_marabou
from linna.basis_finder import VarianceBasisFinder
from linna.network import Network
from linna.utils import load_tf_network, nnet_to_torch
import matplotlib.pyplot as plt
import torch

from linna.verification.nnet import NNet

sys.path.append('/home/calvin/Repositories/Marabou')

from maraboupy import Marabou
from maraboupy import MarabouCore

NNET_FILE = "../networks/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
DELTA = 0.1

nnet_network = NNet(filename=NNET_FILE)
print(f"Num layers: {nnet_network.weights}")
sequential = nnet_to_torch(nnet_network)

x = np.array([0.61, 0.36, 0.0, 0.0, -0.24])
target_cls = nnet_network.evaluate_network(x).argmin()

options = Marabou.createOptions(verbosity=0)

# Export network
linna_net = Network(sequential)

bf = VarianceBasisFinder(network=linna_net,
                         io_dict={
                             idx: layer.get_weight().cpu().detach().numpy().T for idx, layer in
                             enumerate(linna_net.layers)
                         })

first, last = 0, len(linna_net.layers) - 1
for layer_idx, layer in enumerate(linna_net.layers):
    if layer_idx not in [first, last]:
        print(f"Abstract layer: {layer_idx}")
        basis = bf.find_basis(layer_idx=layer_idx, basis_size=35)
        layer.basis = basis
        for neuron in layer.neurons:
            if neuron not in basis:
                layer.neuron_to_lower_bound[neuron] = lp_lower_bound(linna_net, layer_idx=layer_idx, neuron=neuron)
                a, b = lp_upper_bound(linna_net, layer_idx=layer_idx, neuron=neuron)
                layer.neuron_to_upper_bound[neuron] = a
                layer.neuron_to_upper_bound_term[neuron] = b

cex = linna_to_marabou(linna_net, x=x, delta=DELTA, target=target_cls)

cex_out = linna_net.forward(torch.Tensor(cex))

print(f"Original out: {linna_net.forward(torch.Tensor(x))}")
print(f"Cex out: {cex_out}")
print(f"Original target cls: {target_cls}")