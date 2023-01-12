import sys, os
import numpy as np
from torchvision import datasets, transforms

from experiments.playground.bounds import lp_lower_bound, lp_upper_bound
from experiments.playground.marabou_network import linna_to_marabou_network
from experiments.playground.marabou_utils import linna_to_marabou
from linna.basis_finder import VarianceBasisFinder
from linna.network import Network
from linna.utils import load_tf_network
import torch

from tests.toy_network import create_toy_network

sys.path.append('/home/calvin/Documents/tools/Marabou')

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

X, y = next(iter(trainloader))
x = X[0].view(-1, 784)[0]

DELTA = 0.05

from maraboupy import Marabou
from maraboupy import MarabouCore

options = Marabou.createOptions(verbosity=0)

# Export network
sequential = load_tf_network(file="../networks/MNIST_5x100.tf")
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
        basis = bf.find_basis(layer_idx=layer_idx, basis_size=[95, 80, 80, 95][layer_idx-1])
        layer.basis = basis
        for neuron in layer.neurons:
            if neuron not in basis:
                layer.neuron_to_lower_bound[neuron] = lp_lower_bound(linna_net, layer_idx=layer_idx, neuron=neuron)
                a, b = lp_upper_bound(linna_net, layer_idx=layer_idx, neuron=neuron)
                layer.neuron_to_upper_bound[neuron] = a
                layer.neuron_to_upper_bound_term[neuron] = b

cex = linna_to_marabou(linna_net, x=x, delta=DELTA, target=y[0].item())
if cex is not None:
    print(y[0].item())
    print(type(cex))
    print(linna_net.forward(torch.tensor(cex.astype("float32"))))
    print("-----------------------------")
    print(linna_net.forward(x))
