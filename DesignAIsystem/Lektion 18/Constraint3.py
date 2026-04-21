import constraint
from constraint import FunctionConstraint

problem = constraint.Problem()

Warehouses = {'Bonn', 'Bordeaux', 'London', 'Paris', 'Rome' }
capacity = [1, 4, 2, 1, 3]
fixed = 30


for store in range(10):
    problem.addVariable(f'store_{store}', [0, 1, 2, 3, 4])


def capacity_ok(*stores):
    for w in range(5):
        if stores.count(w) > capacity[w]:
            return False
    return True

problem.addConstraint(FunctionConstraint(capacity_ok), [f'store_{i}' for i in range(10)])


supplyCost = [
    [20, 24, 11, 25, 30],
    [28, 27, 82, 83, 74],
    [74, 97, 71, 96, 70],
    [2,  55, 73, 69, 61],
    [46, 96, 59, 83,  4],
    [42, 22, 29, 67, 59],
    [1,   5, 73, 59, 56],
    [10, 73, 13, 43, 96],
    [93, 35, 63, 85, 46],
    [47, 65, 55, 71, 95],
]

solutions = problem.getSolutions()

best_cost = float('inf')
best_solution = None

for s in solutions:
    supply_total = 0
    for store in range(10):
        warehouse = s[f'store_{store}']
        supply_total += supplyCost[store][warehouse]
    
    open_warehouses = set(s.values())
    total = supply_total + len(open_warehouses) * fixed
    
    if total < best_cost:
        best_cost = total
        best_solution = s

print(f"Bedste cost: {best_cost}")
print(best_solution)
