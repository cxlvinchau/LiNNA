import sys
import numpy as np

sys.path.append('/home/calvin/Repositories/Marabou')


from maraboupy import Marabou
from maraboupy import MarabouCore

network = Marabou.read_nnet("../../networks/nnet/ACASXU_experimental_v2a_1_1.nnet")
input_vars = network.inputVars[0][0]
output_vars = network.outputVars[0][0]

out = network.evaluateWithMarabou(np.ones(5))
print(out)

network.setLowerBound(input_vars[0], 0.6)
network.setUpperBound(input_vars[0], 0.6798577687)
network.setLowerBound(input_vars[1], -0.5)
network.setUpperBound(input_vars[1], 0.5)
network.setLowerBound(input_vars[2], -0.5)
network.setUpperBound(input_vars[2], 0.5)
network.setLowerBound(input_vars[3], -0.5)
network.setUpperBound(input_vars[3], 0.5)
network.setLowerBound(input_vars[4], -0.5)
network.setUpperBound(input_vars[4], -0.45)
network.setLowerBound(output_vars[0], 3.9911256459)

status, result, stats = network.solve()
print()
print(50*"=")
print(f"Total time: {stats.getTotalTimeInMicro()} microseconds")
