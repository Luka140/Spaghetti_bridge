import numpy as np 
import matplotlib.pyplot as plt
from main import main  

def gLmax(L, Lmax, p=20):
    if L>=Lmax:
        print('L>=Lmax', L)
    return 100*np.exp(L - Lmax)**p 

def gLmin(L, Lmin, p=20):

    return 1/np.exp(L - Lmin)**p 

def gMmax(M, Mmax, p=20):
    return np.exp(M - Mmax)**p 

def gMmin(M, Mmin, p=20):
    return 1/np.exp(M - Mmin)**p 

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
    #print('failure_mass:', failure_mass, 'gLmax:', gLmax(L,Lmax), 'gLmin:', gLmin(L, Lmin), 'gMmax:', gMmax(mass, Mmax), 'gMmin:', gMmin(mass, Mmin))
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
    
    grad = grad/np.linalg.norm(grad)
    
    return grad 

def steespest_ascent(params, lr, tol, x, y, res):
    grad = gradient(optimizer, params)
    res1, res2 = 0, 1
    while np.abs(res1-res2) > tol:
        res1 = optimizer(params)
        params += lr * grad
        res2 = optimizer(params)
        if res2 < res1:
            lr *= 0.9
            params -= lr * grad
        else:
            lr *= 1.2
        grad = gradient(optimizer, params)
        x.append(params[0])
        y.append(params[1])
        res.append(res2)
    return params

chainx = []
chainy = []
result = []
steespest_ascent(params=np.array([2, -0.07]), lr=np.array([1e-2, 1e-3]), tol=1e-6, x=chainx, y=chainy, res=result)

print(len(chainx), len(chainy), len(result))
N = 20
M = 20
f = np.zeros((N, M))

A = np.linspace(1,4, N)
MID = np.linspace(-0.08, -0.01, M)
it = 0 
for a in range(len(A)):
    for mid in range(len(MID)):
        it +=1
        f[a, mid], L, mass = main(A[a], 20, 12, 4, MID[mid])
        if L>Lmax  or mass> Mmax:
            f[a,mid] = 0 
X, Y = np.meshgrid(A, MID)

plt.close()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(chainx, chainy, result)
levels = [400,500,600,700,725, 750, 760, 775,800,825]
ax.contour(X, Y, f, levels=levels)
ax.plot(chainx[-1], chainy[-1], result[-1], marker='o', color='r')
ax.set_xlabel('a')
ax.set_ylabel('mid')
ax.set_zlabel('Objective Function Value')
plt.show()