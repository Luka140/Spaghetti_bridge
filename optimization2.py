import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from main import main  

def gLmax(L, Lmax, p=10):
    if L >= Lmax:
        print('L>=Lmax', L)
    return 100 * np.exp((L - Lmax) / 0.01) ** p 

def gLmin(L, Lmin, p=10):
    return 1 / np.exp(L - Lmin) ** p 

def gMmax(M, Mmax, p=10):
    return 100 * np.exp((M - Mmax)/0.01) ** p 

def gMmin(M, Mmin, p=10):
    return 1 / np.exp(M - Mmin) ** p 

def gNodeConstraint(nodeconstraint, Nodemax, p=20):
    return 100 * np.exp((nodeconstraint - Nodemax)/0.01) ** p

n = 10
mid = 0 
tanWidth = 12
radWidth = 4

Lmax = 0.3
Lmin = 0.03
Mmax = 0.5
Mmin = 0
Nodemax = 0

def optimizer(params, last=False):
    a1, n, tanwidth, radwidth, mid_h = arguments(params)
    failure_mass, L, mass, nodeconstraint = main(a1, n, tanwidth, radwidth, mid_h)
    objective = failure_mass - gLmax(L, Lmax) - gLmin(L, Lmin) - gMmax(mass, Mmax) - gMmin(mass, Mmin) - gNodeConstraint(nodeconstraint, Nodemax)
    if last:
        print('params:', params, 'failure mass:', failure_mass, 'L:', L, 'mass:', mass, 'node constraint:', nodeconstraint, 
              'gLmax:', gLmax(L, Lmax), 'gLmin:', gLmin(L, Lmin), 'gMmax:', gMmax(mass, Mmax), 'gMmin:', gMmin(mass, Mmin), 'gNodeConstraint:', gNodeConstraint(nodeconstraint, Nodemax))
    return objective

def arguments(params):
    return params[0], n, tanWidth, radWidth, params[1]

def objective_function(params):
    # We minimize the negative of the objective function to perform maximization
    return -optimizer(params)

# Initial guess for the parameters
initial_params = np.array([3, -0.07])

# Use scipy.optimize.minimize to find the parameters that maximize the objective function
result = minimize(objective_function, initial_params, method='BFGS')

# Extract the optimized parameters
optimized_params = result.x

# Evaluate the final objective function value
optimizer(optimized_params, last=True)

# Run the main function with the optimized parameters and plot the results
final, _, _, _ = main(optimized_params[0], n, tanWidth, radWidth, optimized_params[1], plotting=True)

print('Final:: params:', optimized_params, ' mass:', final)
N = 50
M = 50
A = np.linspace(0.1, 8, N)
MID = np.linspace(-0.2, 0.2, M)
f = np.load('map.npy')
X, Y = np.meshgrid(A, MID)
#f = np.load('map.npy')
print('params:', optimized_params, 'objective function:', -result.fun)
plt.close()
mesh = np.meshgrid(A, MID)
plt.pcolormesh(mesh[0], mesh[1], f, cmap='coolwarm')
plt.plot(optimized_params[0], optimized_params[1], 'ko')
plt.show()
