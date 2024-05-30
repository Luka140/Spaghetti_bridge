import numpy as np 
import matplotlib.pyplot as plt
from main import main  

def gLmax(L, Lmax):
    if L>=Lmax:
        print('L>=Lmax')
    return np.log(-L + Lmax)

def gLmin(L, Lmin):

    return np.log(L - Lmin)

def gMmax(M, Mmax):
    return np.log(-M + Mmax)

def gMmin(M, Mmin):
    return np.log(M - Mmin)

n = 20
mid = 0 

starting_point = np.array([2, n, 12, 4, mid])

Lmax = 0.3
Lmin = 0.03
Mmax = 0.5
Mmin = 0

def optimizer(params):
    a1, n, tanwidth, radwidth, mid_h = arguments(params)
    failure_mass, L, mass = main(a1, n, tanwidth, radwidth, mid_h)
    print('Length max is', L)
    return failure_mass - gLmax(L,Lmax) - gLmin(L, Lmin) - gMmax(mass, Mmax) -gMmin(mass, Mmin)

def arguments(params):
    return params[0], n, 12, 4, params[1]

def gradient(optimizer, params):
    h = 1e-3
    grad = np.zeros(len(params))
    params_new = np.copy(params)
    for i in range(len(params)):
        params_new[i] += h
        grad[i] = (optimizer(params_new) - optimizer(params))/h
        params_new[i] -= h
    return grad

def steespest_ascent(params, lr, tol):
    grad = gradient(optimizer, params)
    while np.linalg.norm(grad) > tol:
        params += lr * grad
        grad = gradient(optimizer, params)
    return params

N = 30
M = 30
f = np.zeros((N, M))

A = np.linspace(0.2,6, N)
MID = np.linspace(-0.08, 0.0, M)
it = 0 
for a in range(len(A)):
    for mid in range(len(MID)):
        it +=1
        print(it)
        f[a, mid], _,_ = main(A[a], 10, 12, 4, MID[mid])

#PLOT f 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(A, MID)
ax.plot_surface(X, Y, f)

plt.show()

steespest_ascent(params=np.array([2, -0.1]), lr=1e-3, tol=1e-6)