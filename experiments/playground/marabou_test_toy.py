import sys, os
import numpy as np

from experiments.playground.marabou_utils import print_equation, print_relu_constr
from src.network import Network
from src.utils import load_tf_network
import torch

import onnx

from tests.toy_network import create_toy_network

sys.path.append('/home/calvin/Documents/tools/Marabou')

x = np.array([1, 1])

from maraboupy import Marabou
from maraboupy import MarabouCore
options = Marabou.createOptions(verbosity = 0)

# Export network
linna_net = create_toy_network()
torch.onnx.export(linna_net.torch_model, torch.tensor(x).float(), "tmp/toy_network.onnx")


network = Marabou.read_onnx("tmp/toy_network.onnx")

for input_neuron in range(2):
    network.setLowerBound(input_neuron, x[input_neuron])
    network.setUpperBound(input_neuron, x[input_neuron])

vals = network.solve()

for equation in network.equList:
    print_equation(equation)

for relu in network.reluList:
    print_relu_constr(relu)

print(vals)
print(linna_net.forward(torch.tensor(x).float()))