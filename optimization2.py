from scipy.optimize import minimize

# Define the objective function
def objective(x):
    return x[0]**2 + x[1]**2

# Define the constraints
constraints = (
    {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1},  # equality constraint
    {'type': 'ineq', 'fun': lambda x: x[0] - 0.5}       # inequality constraint
)

# Define the bounds
bounds = [(0, None), (0, None)]  # x[0] >= 0, x[1] >= 0

# Initial guess
x0 = [0.5, 0.5]

# Perform optimization
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
print(result)