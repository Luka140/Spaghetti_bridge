import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def StiffMat(Area, I, E, leng):  # define stiffness matrix

    Ktot = np.zeros([6, 6])

    # the truss part
    k = E * Area / leng

    Ktot[0, 0] = k
    Ktot[3, 3] = k
    Ktot[0, 3] = -k
    Ktot[3, 0] = -k

    # the beam part
    k = E * I / leng ** 3;

    Kbeam = np.zeros([4, 4])
    Kbeam[0, 0] = 12 * k;
    Kbeam[0, 1] = 6 * leng * k;
    Kbeam[1, 0] = 6 * leng * k;
    Kbeam[1, 1] = 4 * leng ** 2 * k;

    Kbeam[2, 0] = -12 * k;
    Kbeam[0, 2] = -12 * k;
    Kbeam[0, 3] = 6 * leng * k;
    Kbeam[3, 0] = 6 * leng * k;

    Kbeam[1, 2] = -6 * leng * k;
    Kbeam[2, 1] = -6 * leng * k;
    Kbeam[2, 2] = 12 * k;
    Kbeam[3, 3] = 4 * leng ** 2 * k;

    Kbeam[3, 1] = 2 * leng ** 2 * k;
    Kbeam[1, 3] = 2 * leng ** 2 * k;
    Kbeam[3, 2] = -6 * leng * k;
    Kbeam[2, 3] = -6 * leng * k;

    Ktot[1:3, 1:3] = Kbeam[0:2, 0:2];
    Ktot[4:6, 4:6] = Kbeam[2:4, 2:4];
    Ktot[1:3, 4:6] = Kbeam[0:2, 2:4];
    Ktot[4:6, 1:3] = Kbeam[2:4, 0:2];

    return Ktot


def assemble_K(E, areas, lengths, moi, connections):
    n_nodes = lengths.size + 1
    K_global = np.zeros((n_nodes * 3, n_nodes * 3))
    for el_idx, con1, con2 in enumerate(connections):
        n1idx, n2idx = con1 - 1, con2 - 1
        # n1, n2 = NodePos[n1idx,:], NodePos[n2idx,:]
        K_local = StiffMat(areas[el_idx], moi[el_idx], E, lengths[el_idx])

        indices = (3 * n1idx, 3 * n1idx + 1, 3 * n1idx + 2,
                   3 * n2idx, 3 * n2idx + 1, 3 * n2idx + 2)
        K_global[np.ix_(indices, indices)] += K_local
    return K_global


def reduce_matrices(K, F, BC):
    K_reduced = np.copy(K)
    F_reduced = np.copy(F)
    for col_idx in range(BC.shape[1]):
        dof_idx = BC[0, col_idx] - 1
        bc_disp = BC[1, col_idx]
        F_reduced = F_reduced - K_reduced[:, dof_idx] * bc_disp
    # open_dof = np.ones(K.shape, dtype=bool)
    # open_dof[np.ix_(BC[0,:] -1, BC[0,:] -1)] = False
    return np.delete(np.delete(K_reduced, BC[0, :] - 1, 0), BC[0, :] - 1, 1), \
        np.delete(F_reduced, BC[0, :] - 1)


def gen_con_matrix(element_count):
    con_mat = np.matrix([[i, i + 1] for i in range(1, element_count + 1)])
    return con_mat

def gen_force_vector():
    raise NotImplementedError


def numerical_solution(u_el, l_el, x):
    N1_T = (l_el ** 3 - 3 * l_el * x ** 2 + 2 * x ** 3) / l_el ** 3
    N2_T = (l_el ** 2 * x - 2 * x ** 2 * l_el + x ** 3) / l_el ** 2
    N3_T = (3 * x ** 2 * l_el - 2 * x ** 3) / l_el ** 3
    N4_T = (x ** 3 - x ** 2 * l_el) / l_el ** 2
    v = N1_T * u_el[1] + N2_T * u_el[2] + N3_T * u_el[4] + N4_T * u_el[5]

    N1_A = 1 - x / l_el
    N2_A = x / l_el
    u = N1_A * u_el[0] + N2_A * u_el[3]
    return u, v


def simulate_frame(E, n_elem, a, b, qx, w_frame, h_frame, l_frame, BC):
    con = gen_con_matrix(n_elem)
    lengths = np.array([l_frame / n_elem for i in range(n_elem)])
    moi = np.array([w_frame * h_frame ** 3 / 12 for i in range(n_elem)])
    areas = np.array([h_frame * w_frame for i in range(n_elem)])

    force = gen_force_vector()
    K = assemble_K(E, areas, lengths, moi, con)
    K_red, F_red = reduce_matrices(K, force, BC)
    u_global = np.linalg.solve(K_red, F_red)

    #  make this less disgusting later, this is not very robust
    u = np.zeros((n_elem + 1) * 3)
    u[BC[0, :] - 1] = BC[1, :]
    u[[idx for idx in range((n_elem + 1) * 3) if idx + 1 not in BC[0, :]]] = u_global

    reactions = K @ u - force

    return u, reactions

