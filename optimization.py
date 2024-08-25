import numpy as np
import matplotlib.pyplot as plt
from main import main  

def gLmax(L, Lmax, p=10):
    if L >= Lmax:
        print('L>=Lmax', L)
    return 1 * np.exp((L - Lmax) / 0.01) ** p 

def gLmin(L, Lmin, p=10):
    return 1 / np.exp(L - Lmin) ** p 

def gMmax(M, Mmax, p=10):
    return 1 * np.exp(M - Mmax) ** p 

def gMmin(M, Mmin, p=10):
    return 1 / np.exp(M - Mmin) ** p 

# def gNodeConstraint(nodeconstraint, Nodemax, p=10):
#     return 1 * np.exp(nodeconstraint - Nodemax) ** p

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
    objective = failure_mass - gLmax(L, Lmax) - gLmin(L, Lmin) - gMmax(mass, Mmax) - gMmin(mass, Mmin) 
    if last:
        print('params:', params, 'failure mass:', failure_mass, 'L:', L, 'mass:', mass, 'node constraint:', nodeconstraint, 
              'gLmax:', gLmax(L, Lmax), 'gLmin:', gLmin(L, Lmin), 'gMmax:', gMmax(mass, Mmax), 'gMmin:', gMmin(mass, Mmin), 'gNodeConstraint:')
    return objective

def arguments(params):
    return params[0], n, tanWidth, radWidth, params[1]

def gradient(optimizer, params):
    h = 1e-3  # Larger step size for faster calculations
    grad = np.zeros(len(params))
    f_val = optimizer(params)
    for i in range(len(params)):
        params_new = np.copy(params)
        params_new[i] += h
        grad[i] = (optimizer(params_new) - f_val) / h
    grad /= np.linalg.norm(grad)  # Normalize gradient
    return grad 

def momentum_optimizer(params, initial_lr, tol, x, y, res, gamma=0.9, max_iter=500):
    lr = initial_lr
    res1, res2 = 0, 1
    res1 = optimizer(params)
    velocity = np.zeros_like(params)
    count = 0
    converged  = False
    while not converged and count < max_iter:
        count += 1
        res1 = optimizer(params)

        grad = gradient(optimizer, params)
        velocity = gamma * velocity + lr * grad
        params += velocity

        res2 = optimizer(params)

        value = np.abs((res2 - res1)/res2)/100
        
        if value < 1:
            lr = lr * (1 - value)

        if np.abs(res2 - res1) < tol:
            converged = True

        x.append(params[0])
        y.append(params[1])
        res.append(res2)
        print(params, 'learning rate:', lr, 'objective function:', res2)
    optimizer(params, last=True)
    print('Iterations:', count)
    return params

chainx = []
chainy = []
result = []
lr = 0.001
params = momentum_optimizer(params=np.array([3, -0.07]), initial_lr=lr, tol=1e-5, x=chainx, y=chainy, res=result)
final, _, _, _ = main(params[0], n, tanWidth, radWidth, params[1], plotting=True)

N = 50
M = 50
A = np.linspace(0.1, 8, N)
MID = np.linspace(-0.2, 0.2, M)
X, Y = np.meshgrid(A, MID)
f = np.load('map.npy')
print('params:', params, 'objective function:', result[-1])
plt.close()
plt.pcolormesh(X, Y, f, cmap='viridis')
plt.plot(chainx[-1], chainy[-1], 'ko')
plt.plot(chainx, chainy, 'b-')
plt.show()
