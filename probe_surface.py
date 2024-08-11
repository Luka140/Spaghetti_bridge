from main import main
from simplified_optimization import objective
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib
from plotting import plot_color_contour


if __name__ == '__main__':
    lr = 1e-2
    step = 1e-5
    # Min length, max length, max mass
    l_min, l_max, m_max = 0.03, 0.3, 0.5
    constraints = [l_min, l_max, m_max]

    n = 10
    tanwidth = 12
    radwidth = 4

    steps = 600
    a_range = np.linspace(0, 5, steps)
    h_range = np.linspace(-0.2, -0.001, steps)

    obj_values = np.zeros((steps, steps))

    for i, a in enumerate(a_range):
        for j, h in enumerate(h_range):
            carried_mass, bridge_mass, min_length, max_length = main(a1=a, n=n, tanwidth=tanwidth, radwidth=radwidth, mid_h=h)
            obj = objective(main, [a, n, tanwidth, radwidth, h], max_constraints=constraints)
            obj_values[i, j] = obj

        print(i)

    mesh_a, mesh_h = np.meshgrid(a_range, h_range)

    version = '2'
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
