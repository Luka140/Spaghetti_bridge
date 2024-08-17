import numpy as np 
import copy
from main import main
from logger import logger
from plotting import plot_color_contour
import matplotlib.pyplot as plt
from generate_bridge import generate_bridge
import scipy.optimize as opt

def gLmax(L, Lmax, p=100., p2=0.00001):
    return p * np.exp((L - Lmax)/p2)

def gLmin(L, Lmin, p=100., p3=100.):
    return p / np.exp(p3*(L - Lmin))


def gMmax(M, Mmax, p=100., p2=0.00001):
    return p * np.exp((M - Mmax)/p2)


def objective(analysis_function, params, max_constraints, **kwargs):
    p1 = kwargs.get('p1', None)
    if p1 is None:
        p1 = 10
    p2 = kwargs.get('p2', None)
    if p2 is None:
        p2 = 0.01

    failure_mass, mass_bridge, min_length, max_length = analysis_function(*params, **kwargs)
    c_min_length, c_max_length, c_max_mass = max_constraints
    penalties = [gMmax(mass_bridge, c_max_mass, p=p1, p2=p2), gLmin(min_length, c_min_length, p=p1, p3=1/p2), gMmax(max_length, c_max_length, p=p1, p2=p2)]

    obj = failure_mass
    obj -= sum(penalties)
    return obj


def gradient(f_analysis, f_obj, params, constraints, h, **kwargs):
    grad = np.zeros(len(params))
    centre_eval = f_obj(f_analysis, params, constraints, **kwargs)
    for i in range(len(params)):
        modification = np.zeros(params.shape)
        modification[i] += h
        eval = f_obj(f_analysis, params + modification, constraints, **kwargs)
        grad[i] = (eval - centre_eval) / h
    return grad, centre_eval


def optimize(analysis_func, objective_func, params, constraints, lr, tol, h, **kwargs):
    scaling_parameter = kwargs.get('scaling_parameter', None)
    if scaling_parameter is None:
        scaling_parameter = 1

    x_prev = copy.deepcopy(params)
    x = copy.deepcopy(params)
    x_history = [x]

    objective_prev = -1e10

    convergence_steps = 0
    converged = False

    try:
        while not converged:
            convergence_steps += 1

            grad, objective = gradient(f_analysis=analysis_func,
                                       f_obj=objective_func,
                                       params=x,
                                       constraints=constraints,
                                       h=h, **kwargs)

            direction = grad / np.linalg.norm(grad)
            x = x_prev + direction * lr

            if objective < objective_prev:
                lr *= 0.2
            else:
                lr *= 1.02

            if abs((objective - objective_prev) / objective) <= tol:
                print(f"Iteration {convergence_steps} - objective value {objective} - parameter values: {x * np.array([scaling_parameter, 1])}")
                converged = True

            # if np.linalg.norm(grad) <= tol:
            #     print(f"Iteration {convergence_steps} - objective value {objective} - parameter values: {x}")
            #     converged = True

            x_history.append(copy.deepcopy(x))
            objective_prev = objective
            x_prev = x

            if convergence_steps % 10 == 0:
                print(f"Iteration {convergence_steps} - objective value {objective} - parameter values: {x * np.array([scaling_parameter, 1])} - stepsize {lr}")

    except KeyboardInterrupt:
        converged = True

    return x, convergence_steps, objective, np.array(x_history)


def simplified_analysis(*params, **kwargs):
    scaling_parameter = kwargs.get('scaling_parameter', None)
    if scaling_parameter is None:
        scaling_parameter = 1

    pars = [params[0]*scaling_parameter,        # a1
            10,                                 # N
            12,                                 # tanwidth
            4,                                  # radwidth
            params[1]]                          # midheight

    return main(*pars)


if __name__ == '__main__':
    scaling_parameter = 5
    initial_params = np.array([1.8 / scaling_parameter, -0.02])
    # initial_params = np.array([4 / scaling_parameter, -0.175])
    lr = 5e-2
    tol = 1e-5
    h = 1e-8

    # Penalty function parameters
    p1 = 5
    p2 = 0.01

    # Min length, max length, max mass
    constraints = [0.03, 0.3, 0.5]

    version = '2'
    dir_a = f"stored_meshes/avals{version}.npy"
    dir_h = f"stored_meshes/hvals{version}.npy"
    dir_obj = f"stored_meshes/objvals{version}.npy"
    mesh_a = np.load(dir_a)
    mesh_h = np.load(dir_h)
    mesh_obj = np.load(dir_obj)

    plot_color_contour(mesh_a, mesh_h, mesh_obj)

    optimal, steps, objective_val, x_history = optimize(simplified_analysis, objective, initial_params, constraints, lr, tol, h,
                                                        scaling_parameter=scaling_parameter,
                                                        p1=p1,
                                                        p2=p2)
    carried_mass, bridge_mass, min_length, max_length = simplified_analysis(*optimal, scaling_parameter=scaling_parameter)

    logger("log_files_simplified", "optimization",
           initial_parameters=str(initial_params * np.array([scaling_parameter, 1])),
           constraints=str(constraints),
           learning_rate=lr,
           tolerance=tol,
           steps=steps,
           optimal_solution=optimal * np.array([scaling_parameter, 1]),
           objective_value=objective_val,
           carried_mass=carried_mass,
           bridge_mass=bridge_mass,
           minimum_spaghetti_length=min_length,
           maximum_spaghetti_length=max_length,
           p1=p1,
           p2=p2
           )

    plt.plot(x_history[:,0]*scaling_parameter, x_history[:,1], color='red', alpha=0.5, linestyle='dashed')
    plt.plot(x_history[:, 0]*scaling_parameter, x_history[:, 1], color='red', marker='o', alpha=0.3)
    plt.plot(x_history[-1, 0] * scaling_parameter, x_history[-1, 1], color='black', marker='x')
    plt.show()

    generate_bridge(a1=x_history[-1,0]*scaling_parameter, N=10,tanWidth=12, radWidth=4, mid_height=x_history[-1,1], plotting=True)
