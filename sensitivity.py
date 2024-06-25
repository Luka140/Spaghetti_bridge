import numpy as np 
from main import main
import matplotlib.pyplot as plt

# sensitivity analysis
def objective_function(a1, n, tanwidth, radwidth, mid_h):
    failure_mass,_,mass = main(a1, n, tanwidth, radwidth, mid_h)
    return failure_mass

N = 20
A1 = np.linspace(1,4,N)
MID = np.linspace(-0.08, -0.01, N)

f = np.zeros((N,N))

#sensitivity analysis
# Choose optimum h 


