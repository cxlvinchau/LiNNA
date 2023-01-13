import sys, os
import numpy as np
from torchvision import datasets, transforms
from experiments.playground.marabou_utils import print_equation, print_relu_constr, linna_to_marabou
from linna.network import Network
from linna.utils import load_tf_network
import torch

from tests.toy_network import create_toy_network

sys.path.append('/home/calvin/Documents/tools/Marabou')

from maraboupy import Marabou
from maraboupy import MarabouCore

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

X, y = next(iter(trainloader))
x = X[0].view(-1, 784)[0]

DELTA = 0.04

# Export network
sequential = load_tf_network(file="../networks/MNIST_5x100.tf")
linna_net = Network(sequential)
torch.onnx.export(linna_net.torch_model, X[0].view(-1, 784), "tmp/model.onnx")

sequential = load_tf_network(file="../networks/MNIST_3x100.tf")
linna_net = Network(sequential)

network = Marabou.read_onnx("tmp/model.onnx")

for input_neuron in range(784):
    network.setLowerBound(input_neuron, x[input_neuron] - DELTA)
    network.setUpperBound(input_neuron, x[input_neuron] + DELTA)

target_cls = y[0].item()
output_vars = network.outputVars[0][0]
target_var = output_vars[target_cls]


# Verify robustness

def robustness():
    for cls in range(10):
        if cls != target_cls:
            network.addInequality(vars=[output_vars[cls], output_vars[target_cls]], coeffs=[-1, 1], scalar=0)


def range_prop():
    network.addInequality(vars=[output_vars[target_cls]], coeffs=[-1], scalar=10)

robustness()


# var_sum = 0
# for equation in network.equList:
#     print_equation(equation)
#     var_sum += len(equation.addendList)
#
# for relu in network.reluList:
#     print_relu_constr(relu)

status, result, stats = network.solve()

print("================================")
print(f"TARGET CLASS = {target_cls}")
print(f"Total time: {stats.getTotalTimeInMicro()} ms")
print(status.upper())
