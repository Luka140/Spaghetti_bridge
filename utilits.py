import numpy as np

def computeMass(elementdata, density):
    return np.sum(elementdata[:, 1]*elementdata[:, 2]*density)
    