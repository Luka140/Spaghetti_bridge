import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from scipy.optimize import bisect, fsolve


def generate_bridge(a1, N, tanWidth, radWidth, mid_height, plotting=False):
    # a1 = 2
    # N = 10
    # tanWidth = 4
    # radWidth = 1
    midpoint = [0,mid_height]
    spaghetti_area = np.pi*3e-3**2/4
    spaghetti_diameter = np.sqrt(4*spaghetti_area/np.pi)

    span = 0.25
    b1 = a1*span**2

    # define parabola

    def f(x):
        return -a1*x**2 + b1

    def momentOfInertia(Width):
        inertia = 0
        # even case
        if Width%2 == 0:
            for i in range(int(Width/2)):
                inertia += 2*Width * (np.pi/64 * spaghetti_diameter**4 + ((i+1)*spaghetti_diameter  - spaghetti_diameter/2)**2 * spaghetti_area)
            # print(1/12 * (Width*spaghetti_diameter)**4)
            # print(inertia)
            return inertia

        # odd case
        inertia = Width * (np.pi/64 * spaghetti_diameter**4)
        for i in range(int(Width/2)):
            inertia += 2*Width * (np.pi/64 * spaghetti_diameter**4 + ((i+1)*spaghetti_diameter)**2 * spaghetti_area)
        # print(1/12 * (Width*spaghetti_diameter)**4)
        # print(inertia)
        return inertia

    # def trasform(theta):
    #     return np.array([[np.cos(theta), -np.sin(theta), 0 ,0]
    #                      [np.sin(theta), np.cos(theta), 0, 0 ],
    #                      [0, 0, np.cos(theta), -np.sin(theta)],
    #                      [0, 0, np.sin(theta), np.cos(theta) ]])

    x = np.linspace(0, 0.25, 20)
    y = f(x)


    alpha_start = np.arctan(-midpoint[1]/span)
    alphas = np.linspace(alpha_start, np.pi/2, N)
    nodes = np.zeros((N, 2))

    for i, alpha in enumerate(alphas):
        if np.abs(alpha - np.pi/2) < 1e-6:
            nodes[i, 0] = 0
            nodes[i, 1] = f(0)
            if plotting:
                plt.plot([midpoint[0], nodes[i,0]], [midpoint[1], nodes[i,1]], 'r')
            continue
        y1 = lambda x: np.tan(alpha) * x + midpoint[1]
        nodes[i, 0] = fsolve(lambda x: y1(x) - f(x), 3)
        nodes[i, 1] = f(nodes[i, 0])
        if plotting:
            plt.plot([midpoint[0], nodes[i,0]], [midpoint[1], nodes[i,1]], 'r')

    for i in range(len(nodes)-1):
        if plotting:
            plt.text(nodes[i,0]+ 1e-4, nodes[i,1]+ 1e-4, str(i+1))
            plt.plot([nodes[i,0], nodes[i+1,0]], [nodes[i,1], nodes[i+1,1]], color='black')
            if i == len(nodes)-2:
                plt.text(nodes[i+1,0]+ 1e-4, nodes[i+1,1]+ 1e-4, str(i+2))
    # Connectivity matrix
    radial = np.vstack((np.zeros(N, dtype=np.integer), np.arange(1, N+1, dtype=np.integer))).T
    tangential = np.vstack((np.arange(1, N,dtype=np.integer), np.arange(2, N+1, dtype=np.integer))).T
    connection_matrix = np.vstack((radial, tangential))

    # Element data
    idx = np.arange(0, connection_matrix.shape[0])

    radial_lengths = np.sqrt((nodes[:, 0] - midpoint[0])**2 + (nodes[:, 1] - midpoint[1])**2)
    radial_areas = radWidth*radWidth*spaghetti_area * np.ones(N)
    radial_inertia = np.ones(N)*momentOfInertia(radWidth)

    tangential_lengths = np.sqrt((nodes[1:, 0] - nodes[:-1, 0])**2 + (nodes[1:, 1] - nodes[:-1, 1])**2)
    tangential_areas = tanWidth*tanWidth*spaghetti_area * np.ones(N-1)
    tangential_inertia = np.ones(N-1)*momentOfInertia(tanWidth)

    lengths = np.hstack((radial_lengths, tangential_lengths))
    areas = np.hstack((radial_areas, tangential_areas))
    inertia = np.hstack((radial_inertia, tangential_inertia))

    elements = np.vstack((idx, lengths, areas, inertia)).T
    nodes = np.vstack((midpoint, nodes))
    # print(nodes)
    # print(elements)

    if plotting:
        plt.text(midpoint[0], midpoint[1]+ 1e-4, str(0))
        plt.scatter(nodes[:, 0], nodes[:, 1], c='g')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x - width [m]')
        plt.ylabel('y - height [m]')
        plt.tight_layout(pad=0.5)

        plt.show()

    return nodes, elements, connection_matrix



