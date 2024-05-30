import numpy as np 

def gLmax(L, Lmax):
    return np.ln(-L + Lmax)

def gLmin(L, Lmin):
    return np.ln(L - Lmin)

def gMmax(M, Mmax):
    return np.ln(-M + Mmax)

def gMmin(M, Mmin):
    return np.ln(M - Mmin)



def F(params):
    return params[0] * params[1]**2

def gradient(F, params):
    h = 1e-5
    grad = np.zeros(len(params))
    for i in range(len(params)):
        params[i] += h
        grad[i] = (F(params) - F(params)) / h
        params[i] -= h
    return grad

def steespest_descent(F, params, lr, tol):
    grad = gradient(F, params)
    while np.linalg.norm(grad) > tol:
        params -= lr * grad
        grad = gradient(F, params)
    return params

print(steespest_descent(F, [1, 0.25], 1e-3, 1e-6))