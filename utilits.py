import numpy as np

def computeMass(elementdata, density):
    print('lengths:', elementdata[:, 1])
    print('areas:', elementdata[:, 2])
    print('volumes:', elementdata[:, 1]*elementdata[:, 2])
    return np.sum(elementdata[:, 1]*elementdata[:, 2]*density)
    