import constraint
from constraint import FunctionConstraint
map_coloring = constraint.Problem()
map_coloring.addVariables(['NA','E','EA','C','SA','M'],['r','g','b'])

def not_equal(a,b):
    return a != b

map_coloring.addConstraint(FunctionConstraint(not_equal),['NA','E'])
map_coloring.addConstraint(FunctionConstraint(not_equal),['NA','EA'])
map_coloring.addConstraint(FunctionConstraint(not_equal),['NA','C'])
map_coloring.addConstraint(FunctionConstraint(not_equal),['E','EA'])
map_coloring.addConstraint(FunctionConstraint(not_equal),['EA','C'])
map_coloring.addConstraint(FunctionConstraint(not_equal),['EA','SA'])
map_coloring.addConstraint(FunctionConstraint(not_equal),['C','SA'])

print(map_coloring.getSolution())

#Det virker

