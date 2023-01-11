import sys

sys.path.append('/home/calvin/Documents/tools/Marabou')

from maraboupy import MarabouCore


def print_equation(equation):
    l = [f"{str(c)}*x{v}" for c, v in equation.addendList]

    if equation.EquationType == MarabouCore.Equation.EQ:
        eq_type = "="
    elif equation.EquationType == MarabouCore.Equation.LE:
        eq_type = "<="
    elif equation.EquationType == MarabouCore.Equation.LE:
        eq_type = "=<"

    print(f"{' + '.join(l)} {eq_type} {equation.scalar}")


def print_relu_constr(t):
    print(f"x{t[1]} = ReLU({t[0]})")


def linna_to_marabou() -> None:
    pass
