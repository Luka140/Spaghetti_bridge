import matplotlib.pyplot as plt
import numpy as np


def plot_color_contour(mesh_x, mesh_y, obj_vals):
    # Contour plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # contour_levels = list(np.linspace(np.min(obj_values), 0, 5))[2:]
    contour_levels = np.linspace(0, 700, 15)
    cp = ax.contour(mesh_x, mesh_y, obj_vals.T, levels=contour_levels, colors='fuchsia', alpha=0.5)
    pm = ax.pcolormesh(mesh_x, mesh_y, obj_vals.T, vmin=-200, vmax=np.max(obj_vals))
    fig.colorbar(pm)
    plt.clabel(cp, inline=True, fontsize=10)
    plt.xlabel('$a$ value of the arc-parabola $y=-ax^2+b$')
    plt.ylabel('midpoint height (m)')
    plt.title('Contour Plot of Objective Function')
    plt.tight_layout(pad=0.5)
    # ax.imshow(interpolation='none')
    # plt.show()
