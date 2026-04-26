import constraint

problem = constraint.Problem()

vehicle_domain = ['r', 'ry', 'g', 'y']
pedestrian_domain = ['r', 'g']

for i in range(1, 5):
    problem.addVariable(f'V{i}', vehicle_domain)
    problem.addVariable(f'P{i}', pedestrian_domain)

vehicle_tuples = [
                ('r','r','g','g'),
                ('ry','r','y','r'),
                ('g','g','r','r'),
                ('y','r','ry','r')]

def check(vi,pi,vj,pj):
    return (vi,pi,vj,pj) in vehicle_tuples

for i in range(1,5):
    j = (i % 4) + 1
    problem.addConstraint(constraint.FunctionConstraint(check),[f'V{i}', f'P{i}', f'V{j}', f'P{j}'])
    
    
solutions = problem.getSolutions()    
print(f"Antal løsniner: {len(solutions)}")
for s in solutions:
    print(s)