import sys

from linna.network import Network

sys.path.append('/home/calvin/Repositories/Marabou')

from maraboupy import MarabouCore
from maraboupy import Marabou

import numpy as np
OPTIONS = Marabou.createOptions()


def print_equation(equation):
    l = [f"{type(c)}*x{v}" for c, v in equation.addendList]

    if equation.EquationType == MarabouCore.Equation.EQ:
        eq_type = "="
    elif equation.EquationType == MarabouCore.Equation.LE:
        eq_type = "<="
    elif equation.EquationType == MarabouCore.Equation.LE:
        eq_type = "=<"

    print(f"{' + '.join(l)} {eq_type} {type(equation.scalar)}")


def print_relu_constr(t):
    print(f"x{t[1]} = ReLU({t[0]})")


def get_input_query(network: Network):
    ipq = MarabouCore.InputQuery()
    # Setup variables
    variables = []
    # Input variables
    input_variables = list(range(network.layers[0].get_weight().shape[1]))
    variables += input_variables
    free_variables = []

    def get_new_variable():
        variables.append(len(variables))
        return len(variables) - 1

    # Setup equations
    output_variables = []
    variable_pairs = []

    pre_activation_variables = input_variables
    post_activation_variables = input_variables
    old_post_activations = None
    for idx, layer in enumerate(network.layers):
        weight = layer.get_weight()
        bias = layer.get_bias()
        new_post_activation_variables = []
        new_pre_activation_variables = []
        for neuron in layer.neurons:
            var = get_new_variable()
            # If neuron is in the basis
            if neuron not in layer.neuron_to_lower_bound or True:
                equation = MarabouCore.Equation()
                new_pre_activation_variables.append(var)
                for neuron_idx, in_var in enumerate(post_activation_variables):
                    equation.addAddend(weight[neuron][neuron_idx], in_var)
                equation.addAddend(-1, var)
                equation.setScalar(-bias[neuron])
                if idx < len(network.layers) - 1:
                    relu_var = get_new_variable()
                    new_post_activation_variables.append(relu_var)
                    MarabouCore.addReluConstraint(ipq, var, relu_var)
                else:
                    output_variables.append(var)
                ipq.addEquation(equation)
            else:
                new_pre_activation_variables.append(var)
                new_post_activation_variables.append(var)

            # Sanity check - All post activation variables are greater equals zero
            # for var in post_activation_variables:
            #     equation = MarabouCore.Equation(MarabouCore.Equation.GE)
            #     equation.addAddend(1, var)
            #     equation.setScalar(0)
            #     inputQuery.addEquation(equation)

        if idx < len(network.layers) - 1:
            old_post_activations = post_activation_variables
            post_activation_variables = new_post_activation_variables
            pre_activation_variables = new_pre_activation_variables
            if idx == 0:
                debug_postactivations = post_activation_variables
            if idx == 1:
                debug_preactivations = pre_activation_variables

        # Test bounds
        for neuron, relu_var in zip(layer.neurons, post_activation_variables):
            if neuron in layer.neuron_to_lower_bound and idx < len(network.layers) - 1:
                # Lower bound
                equation = MarabouCore.Equation(MarabouCore.Equation.LE)
                for basis_idx, basis_neuron in enumerate(layer.basis):
                    equation.addAddend(layer.neuron_to_lower_bound[neuron][basis_idx],
                                       pre_activation_variables[basis_neuron])
                equation.addAddend(-1, post_activation_variables[neuron])
                equation.setScalar(0)
                ipq.addEquation(equation)

                # Upper bound
                equation = MarabouCore.Equation(MarabouCore.Equation.GE)
                for basis_idx, basis_neuron in enumerate(layer.basis):
                    equation.addAddend(layer.neuron_to_upper_bound[neuron][basis_idx],
                                       post_activation_variables[basis_neuron])

                for neuron_idx, in_var in enumerate(old_post_activations):
                    equation.addAddend(layer.neuron_to_upper_bound_term[neuron][neuron_idx], in_var)

                equation.addAddend(-1, post_activation_variables[neuron])
                equation.setScalar(-layer.neuron_to_upper_bound_term[neuron][-1])
                ipq.addEquation(equation)

                # Debug
                # lower bound
                debug_var_lb = get_new_variable()
                # Greater equals zero due to ReLU
                equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
                for var_idx, in_var in enumerate(old_post_activations):
                    equation.addAddend(layer.neuron_to_lower_bound_alt[neuron][0][var_idx], in_var)
                equation.addAddend(-1, debug_var_lb)
                equation.setScalar(-1 * layer.neuron_to_lower_bound_alt[neuron][1])
                ipq.addEquation(equation)
                # upper bound
                debug_var_ub = get_new_variable()

                equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
                for var_idx, in_var in enumerate(old_post_activations):
                    assert np.all(layer.neuron_to_upper_bound_alt[neuron][0] >= 0)
                    equation.addAddend(layer.neuron_to_upper_bound_alt[neuron][0][var_idx], in_var)

                equation.addAddend(-1, debug_var_ub)
                print(layer.neuron_to_upper_bound_alt[neuron][1])
                equation.setScalar(-layer.neuron_to_upper_bound_alt[neuron][1])
                ipq.addEquation(equation)
                variable_pairs.append((debug_var_lb, post_activation_variables[neuron], debug_var_ub))

                free_variables.append(post_activation_variables[neuron])

    ipq.setNumberOfVariables(len(variables))

    i = 0
    for var in input_variables + free_variables:
        ipq.markInputVariable(var, i)
        if var in free_variables:
            ipq.setLowerBound(var, 0)
        i += 1

    i = 0
    for output_var in output_variables:
        ipq.markOutputVariable(output_var, i)
        i += 1

    print(f"NUMBER of Variables: {len(variables)}")
    print(f"Out vars: {len(output_variables)}")

    return ipq, input_variables, output_variables, variable_pairs


def evaluate_local_robustness(network: Network, x, delta, target_cls):
    ipq, input_variables, output_variables, pairs = get_input_query(network)

    for input_var in input_variables:
        ipq.setLowerBound(input_var, x[input_var] - delta)
        ipq.setUpperBound(input_var, x[input_var] + delta)

    target_var = output_variables[target_cls]

    for idx, output_var in enumerate(output_variables):
        if idx != target_cls:
            MarabouCore.addMaxConstraint(ipq, set([output_var, target_var]), output_var)
            exitCode, vals, stats = MarabouCore.solve(ipq, options=OPTIONS)
            print(80*"=")
            print(f"Query for cls {idx}")
            print(exitCode)
            print(f"Total time: {stats.getTotalTimeInMicro()} microseconds")
            if exitCode == "sat":
                for cls, var in enumerate(output_variables):
                    print(f"Output {cls}: {vals[var]}")

                print(80 * "-")
                print("Pair differencce")
                diff = 0
                for lb, var, ub in pairs:
                    print(f"{max(vals[lb], 0)} <= {vals[var]} <= {vals[ub]}")
                    # assert max(vals[lb], 0) <= vals[var] <= vals[ub]
                    diff += vals[ub] - max(vals[lb], 0)
                print(f"DIFF SUM: {diff}")
                print(80 * "=" + "\n")
                return vals, stats, idx

            print(80*"=" + "\n")

    return None, None, None

