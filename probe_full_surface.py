from full_optimization_nested import objective, full_analysis
from main import main
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib
from plotting import plot_color_contour


if __name__ == '__main__':


    # Min length, max length, max mass
    l_min, l_max, m_max = 0.03, 0.3, 0.5
    constraints = [l_min, l_max, m_max]

    steps = 5
    # Params:
    # a, N, tanwidth, radwidth, height
    a_range = np.linspace(0, 5, steps)
    n_min, n_max, stepsize = 2, 40, 3
    n_range = range(n_min, n_max, stepsize)
    tw_range = np.linspace(1, 20, steps)
    rw_range = np.linspace(1, 20, steps)
    h_range = np.linspace(-0.2, -0.001, steps)

    obj_values = np.zeros((steps,(n_max-n_min), steps, steps, steps))

    for i_a, a in enumerate(a_range):
        for i_n, n in enumerate(n_range):
            for i_tw, tw in enumerate(tw_range):
                for i_rw, rw in enumerate(rw_range):
                    for i_h, h in enumerate(h_range):
                        carried_mass, bridge_mass, min_length, max_length = main(a1=a, n=n, tanwidth=tw, radwidth=rw, mid_h=h)
                        obj = carried_mass
                        if carried_mass > constraints[2]:
                            obj = 0
                        if min_length < constraints[0]:
                            obj = 0
                        if max_length > constraints[1]:
                            obj = 0
                        # obj = objective(params=(a, tw, rw, h), N=n, max_constraints=constraints, p1=50, p2=0.005)

                        obj_values[i_a, i_n, i_tw, i_rw, i_h] = obj

            print(f'First nested loop at iteration {i_n} out of {(n_max-n_min)//stepsize}')
        print(f'Outer loop at iteration {i_a} out of {steps}')

    mesh_a, mesh_h = np.meshgrid(a_range, h_range)

    version = 'full'
    dir_a = pathlib.Path("stored_meshes") / f'avals{version}'
    dir_h = pathlib.Path("stored_meshes") / f'hvals{version}'
    dir_obj = pathlib.Path("stored_meshes") / f'objvals{version}'
    np.save(dir_a, mesh_a)
    np.save(dir_h, mesh_h)
    np.save(dir_obj, obj_values)

    plt.close()
    plot_color_contour(mesh_a, mesh_h, obj_values)
    plt.show()

    # generate_bridge(a1=20, N=n,tanWidth=tanwidth, radWidth=radwidth, mid_height=0.07, plotting=True)
