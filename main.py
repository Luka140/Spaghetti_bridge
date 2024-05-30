import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from generate_bridge import generate_bridge


def get_elemnt_k(Area, E, length):
    K = np.zeros((4,4))
    keq =  E * Area / length
    K[0,0], K[2,2] = keq, keq
    K[0,2], K[2,0] = -keq, -keq

    return K


def assemble_k(E, areas, lengths, connections, node_positions):
    n_nodes = node_positions.shape[0]
    K_global = np.zeros((n_nodes * 2, n_nodes * 2))
    for el_idx in range(len(connections)):
        n1idx, n2idx = connections[el_idx, 0], connections[el_idx, 1]
        n1, n2 = node_positions[n1idx, :], node_positions[n2idx, :]
        dir_vec = n2 - n1
        theta = np.arctan2(dir_vec[1], dir_vec[0])

        K_local = get_elemnt_k(areas[el_idx], E, lengths[el_idx])
        T_mat = get_transform_mat(theta)
        K_transformed = T_mat @ K_local @ T_mat.T
        indices = (2 * n1idx, 2 * n1idx + 1,
                   2 * n2idx, 2 * n2idx + 1)
        K_global[np.ix_(indices, indices)] += K_transformed
    return K_global


def reduce_matrices(K, F, BC):
    K_reduced = np.copy(K)
    F_reduced = np.copy(F)
    for col_idx in range(BC.shape[1]):
        dof_idx = BC[0, col_idx]
        bc_disp = BC[1, col_idx]

        F_reduced = F_reduced - K_reduced[:, dof_idx] * bc_disp
    # open_dof = np.ones(K.shape, dtype=bool)
    # open_dof[np.ix_(BC[0,:] -1, BC[0,:] -1)] = False
    return np.delete(np.delete(K_reduced, BC[0, :], 0), BC[0, :], 1), \
        np.delete(F_reduced, BC[0, :])


def get_transform_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0 ,0],
                     [np.sin(theta), np.cos(theta), 0, 0 ],
                     [0, 0, np.cos(theta), -np.sin(theta)],
                     [0, 0, np.sin(theta), np.cos(theta)]])


# def numerical_solution(u_el, l_el, x):
#     N1_T = (l_el ** 3 - 3 * l_el * x ** 2 + 2 * x ** 3) / l_el ** 3
#     N2_T = (l_el ** 2 * x - 2 * x ** 2 * l_el + x ** 3) / l_el ** 2
#     N3_T = (3 * x ** 2 * l_el - 2 * x ** 3) / l_el ** 3
#     N4_T = (x ** 3 - x ** 2 * l_el) / l_el ** 2
#     v = N1_T * u_el[1] + N2_T * u_el[2] + N3_T * u_el[4] + N4_T * u_el[5]
#
#     N1_A = 1 - x / l_el
#     N2_A = x / l_el
#     u = N1_A * u_el[0] + N2_A * u_Fel[3]
#     return u, v


def simulate_frame(E, node_pos, elements, connections, BC, forces):
    K = assemble_k(E, areas=elements[:, 2], lengths=elements[:, 1], connections=connections, node_positions=node_pos)
    K_red, F_red = reduce_matrices(K, forces, BC)
    u_global = np.linalg.solve(K_red, F_red)

    #  make this less disgusting later, this is not very robust
    u = np.zeros((node_pos.shape[0]) * 2)
    u[BC[0, :] - 1] = BC[1, :]
    u[[idx for idx in range((node_pos.shape[0]) * 2) if idx not in BC[0, :]]] = u_global

    reactions = K @ u - forces
    return u, reactions


def stress_strain(connections, displacements, lengths, node_positions, E):
    """
    Can't be bothered to do it the proper way :)
    """
    strain = np.zeros(connections.shape[0])
    for el_idx, node_idx in enumerate(connections):
        n1, n2 = node_positions[node_idx[0],:], node_positions[node_idx[1],:]
        dir_vec = n2 - n1
        theta = np.arctan2(dir_vec[1], dir_vec[0])
        trans_mat = get_transform_mat(theta)

        dofs = [2*node_idx[0], 2*node_idx[0]+1, 2*node_idx[1], 2*node_idx[1]+1]
        disp_local = trans_mat.T @ (np.array(displacements[dofs])).T
        elongation = disp_local[2:] - disp_local[:2]
        strain[el_idx] = (elongation[0]) / lengths[el_idx]

    stress = strain * E
    return stress, strain


def plot_stress(stress, node_pos, connection_mat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormap = matplotlib.colormaps["jet"]
    norm = plt.Normalize(min(0, min(stress)), max(0, max(stress)))
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=colormap)

    ax.scatter(node_pos[:, 0], node_pos[:, 1], c='black')
    for el_idx, node_idx in enumerate(connection_mat):
        n1, n2 = node_pos[node_idx[0], :], node_pos[node_idx[1], :]
        el_col = scalar_map.to_rgba(stress[el_idx])
        ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color=el_col)

    fig.colorbar(scalar_map, label=r"$\sigma$ [MPa]", ax=ax)
    ax.set_aspect('equal', adjustable='box')

    plt.show()


def main(a1, n, tanwidth, radwidth, midpoint, plotting=False):
    disp_vis_scale = 10
    elasticity_modulus = 10e9

    node_pos, element_data, connection_matrix = generate_bridge(a1=a1,
                                                                N=n,
                                                                tanWidth=tanwidth,
                                                                radWidth=radwidth,
                                                                midpoint=midpoint,
                                                                plotting=plotting)

    # FORMAT OF BCS IS: first row is the degrees of freedom, second row is the constrained displacements
    sym_constraint_idx_top = [(node_pos.shape[0] - 1) * 2]
    # Index 0: horizontal symmetry for the central node
    # sym_constraint_idx_top is the symmetry constraint for the midpoint of the arc
    # 2 and 3 are the dofs for the fully constrained node on the side
    BC_idx = [0, 2, 3] + sym_constraint_idx_top
    BC_disp = [0, 0, 0, 0]
    BC = np.array([BC_idx, BC_disp])

    # TODO: CALCULATE MASS OF BRIDGE

    mass = 10e3  # kg
    force_vector = np.zeros(node_pos.shape[0] * 2)
    force_vector[1] = - mass * 9.81 / 2

    displacements, reactions = simulate_frame(elasticity_modulus, node_pos, element_data, connection_matrix, BC,
                                              force_vector)

    stress, strain = stress_strain(connection_matrix, displacements, lengths=element_data[:, 1],
                                   node_positions=node_pos, E=elasticity_modulus)
    stress_mpa = stress / 10 ** 6

    # print(stress)

    if plotting:
        disp = displacements.reshape(-1, 2)
        new_pos_viz = node_pos + disp * disp_vis_scale
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c='g')
        plt.scatter(new_pos_viz[:, 0], new_pos_viz[:, 1], c='b')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        plot_stress(stress_mpa, node_pos, connection_matrix)


