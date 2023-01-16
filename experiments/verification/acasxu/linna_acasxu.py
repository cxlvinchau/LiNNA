import sys

import torch
import numpy as np

from experiments.verification.acasxu.acas_xu_properties import acas_xu_property_1
from experiments.verification.bounds import lp_lower_bound, lp_upper_bound
from linna.basis_finder import VarianceBasisFinder
from linna.network import Network
from linna.utils import nnet_to_torch
from linna.verification.marabou_utils import get_input_query
from linna.verification.nnet import NNet

sys.path.append('/home/calvin/Repositories/Marabou')

from maraboupy import Marabou
from maraboupy import MarabouCore

nnet = NNet("../../networks/nnet/ACASXU_experimental_v2a_1_1.nnet")
sequential = nnet_to_torch(nnet)
linna_network = Network(sequential)

print(sequential)

out = linna_network.forward(torch.ones((1, 5)))
print(f"Test output: {out}")

bf = VarianceBasisFinder(network=linna_network,
                         io_dict={
                             idx: layer.get_weight().cpu().detach().numpy().T for idx, layer in
                             enumerate(linna_network.layers)
                         })


for layer_idx, layer in enumerate(linna_network.layers[1:-1]):
    print(f"Abstract layer: {layer_idx+1}")
    basis = bf.find_basis(layer_idx=layer_idx+1, basis_size=35)
    layer.basis = basis
    for neuron in layer.neurons:
        if neuron not in basis:
            layer.neuron_to_lower_bound[neuron] = lp_lower_bound(linna_network, layer_idx=layer_idx+1, neuron=neuron)
            a, b = lp_upper_bound(linna_network, layer_idx=layer_idx+1, neuron=neuron)
            layer.neuron_to_upper_bound[neuron] = a
            layer.neuron_to_upper_bound_term[neuron] = b


ipq, input_vars, output_vars = get_input_query(linna_network)

acas_xu_property_1(ipq, input_vars, output_vars)

status, result, stats = MarabouCore.solve(ipq, options=Marabou.createOptions())

print()
print(80*"=")
print(status.upper())
print(f"Total time: {stats.getTotalTimeInMicro()} microseconds")
cex = np.array([result[var] for var in input_vars])
if status == "sat":
    print(f"Cex: {cex}")
    print(f"Cex out: {linna_network.forward(torch.Tensor(cex.astype('float32')))}")
print(80 * "=")
