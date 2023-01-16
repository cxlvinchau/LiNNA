def acas_xu_property_1(o, input_vars, output_vars):
    """
    x0 >= 0.6
    x0 <= 0.6798577687
    x1 >= -0.5
    x1 <= 0.5
    x2 >= -0.5
    x2 <= 0.5
    x3 >= 0.45
    x3 <= 0.5
    x4 >= -0.5
    x4 <= -0.45
    y0 >= 3.9911256459
    """
    o.setLowerBound(input_vars[0], 0.6)
    o.setUpperBound(input_vars[0], 0.6798577687)
    o.setLowerBound(input_vars[1], -0.5)
    o.setUpperBound(input_vars[1], 0.5)
    o.setLowerBound(input_vars[2], -0.5)
    o.setUpperBound(input_vars[2], 0.5)
    o.setLowerBound(input_vars[3], -0.5)
    o.setUpperBound(input_vars[3], 0.5)
    o.setLowerBound(input_vars[4], -0.5)
    o.setUpperBound(input_vars[4], -0.45)
    o.setLowerBound(output_vars[0], 3.9911256459)