import sys, os
import numpy as np
from torchvision import datasets, transforms

from experiments.verification.mnist.mnist_utils import plot_image, plot_cex
from linna.basis_finder import VarianceBasisFinder, PosBasisFinder
from linna.network import Network
from linna.utils import load_tf_network
import matplotlib.pyplot as plt
import torch

from linna.verification.bounds import lp_lower_bound, lp_upper_bound, lp_upper_bound_alternative, \
    lp_lower_bound_alternative
from linna.verification.marabou_utils import get_input_query, evaluate_local_robustness
from tests.toy_network import create_toy_network

sys.path.append('/home/calvin/Documents/tools/Marabou')

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

img_idx = 5

X, y = next(iter(trainloader))
x = X[img_idx].view(-1, 784)[0]
target_cls = y[img_idx].item()

DELTA = 0.2

from maraboupy import Marabou
from maraboupy import MarabouCore

options = Marabou.createOptions(verbosity=0)

# Export network
sequential = load_tf_network(file="../../networks/MNIST_3x100.tf")
linna_net = Network(sequential)

# Abstract network (i.e. compute lower and upper bounds)
bf = PosBasisFinder(network=linna_net,
                    io_dict={
                        idx: layer.get_weight().cpu().detach().numpy().T for idx, layer in
                        enumerate(linna_net.layers)
                    })


def abstract():
    first, last = 0, len(linna_net.layers) - 1
    for layer_idx, layer in enumerate(linna_net.layers):
        if layer_idx not in [first, last]:
            print(f"Abstract layer: {layer_idx}")
            basis = bf.find_basis(layer_idx=layer_idx, basis_size=[95, 90][layer_idx - 1])
            layer.basis = basis
            for neuron in layer.neurons:
                if neuron not in basis:
                    layer.neuron_to_lower_bound[neuron] = lp_lower_bound(linna_net, layer_idx=layer_idx, neuron=neuron)
                    layer.neuron_to_lower_bound_alt[neuron] = lp_lower_bound_alternative(linna_net,
                                                                                         layer_idx=layer_idx,
                                                                                         neuron=neuron)
                    a, b = lp_upper_bound(linna_net, layer_idx=layer_idx, neuron=neuron)
                    layer.neuron_to_upper_bound[neuron] = a
                    layer.neuron_to_upper_bound_term[neuron] = b
                    layer.neuron_to_upper_bound_alt[neuron] = lp_upper_bound_alternative(linna_net, layer_idx=layer_idx,
                                                                                         neuron=neuron)


abstract()

# Evaluate local robustness
result, stats, max_class = evaluate_local_robustness(network=linna_net,
                                                     x=x.cpu().detach().numpy(),
                                                     delta=DELTA,
                                                     target_cls=target_cls)

if result is not None:
    cex = np.array([result[var] for var in range(784)])
    plot_image(x, title="Original image")
    plot_image(cex, title="Counterexample")
    print("Cex output:")
    print(linna_net.forward(torch.Tensor(cex)))
