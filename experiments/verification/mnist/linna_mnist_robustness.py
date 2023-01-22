import sys, os
import numpy as np
from torchvision import datasets, transforms

from experiments.verification.mnist.gurobi_bounds import find_alpha
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

sys.path.append('/home/calvin/Repositories/Marabou')
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

img_idx = 5

X, y = next(iter(trainloader))
x = X[img_idx].view(-1, 784)[0]
target_cls = y[img_idx].item()
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

X, y = next(iter(trainloader))
x = X[img_idx].view(-1, 784)[0]
target_cls = y[img_idx].item()

DELTA = 0.015

from maraboupy import Marabou
from maraboupy import MarabouCore

options = Marabou.createOptions(verbosity=0)

# Export network
sequential = load_tf_network(file="../../networks/MNIST_3x100.tf")
linna_net = Network(sequential)
lb, ub = linna_net.propagate_interval(lb=x.cpu().detach().numpy() - DELTA,
                                      ub=x.cpu().detach().numpy() + DELTA,
                                      layer_idx=0)

# Abstract network (i.e. compute lower and upper bounds)
bf = PosBasisFinder(network=linna_net,
                    io_dict={
                        idx: layer.get_weight().cpu().detach().numpy().T for idx, layer in
                        enumerate(linna_net.layers)
                    })


infeasible = []
def abstract(layer_idx, basis_size):
    layer = linna_net.layers[layer_idx]
    layer.basis = bf.find_basis(layer_idx=layer_idx, basis_size=basis_size)
    for neuron in layer.neurons:
        if neuron not in layer.basis:
            layer.neuron_to_lower_bound[neuron] = lp_lower_bound_alternative(linna_net,
                                                                             layer_idx=layer_idx,
                                                                             neuron=neuron)
            try:
                layer.neuron_to_upper_bound[neuron] = find_alpha(linna_net=linna_net, layer_idx=layer_idx,
                                                                 lb=lb, ub=ub, target_neuron=neuron)
            except:
                infeasible.append(neuron)
                print(f"Could not find combination for {neuron}")


abstract(layer_idx=1, basis_size=90)


print(f"Infeasible neurons (#{len(infeasible)}): {infeasible}")

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
