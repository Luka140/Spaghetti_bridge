import numpy as np 
import copy
from main import main
from logger import logger
from plotting import plot_color_contour
import matplotlib.pyplot as plt
from generate_bridge import generate_bridge
import scipy.optimize as opt

def gLmax(L, Lmax, p=100, p2=0.00001):
    return p * np.exp((L - Lmax)/p2)

def gLmin(L, Lmin, p=100, p3=100):
    return p / np.exp(p3*(L - Lmin))


def gMmax(M, Mmax, p=100, p2=0.00001):
    return p * np.exp((M - Mmax)/p2)


# def gMmin(M, Mmin, p=100):
#     return 100 / np.exp(M - Mmin)**p
#

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
    # print(f"Carrying {failure_mass:2f} kg - bridge mass: {mass_bridge:2f} kg - length range {min_length:2f} {max_length:2f}")
    # print(f"{penalties[0]:2e} - {penalties[1]:2e} - {penalties[2]:2e} ")
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

    ri_prev = 0
    grad_prev = np.zeros(x.shape)
    direction_prev = np.zeros(x.shape)
    velocity = np.zeros(x.shape)

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

            # =========================== Method 1
            # alpha = 0.8
            # ri = alpha * ri_prev + (1 - alpha) * grad * grad
            # dx = lr * grad / (ri**0.5 + 1.e-6)
            # x = x_prev + dx

            # ============================ Method 2
            # alpha = 0.8
            # rho = 0.9
            # ri = alpha * ri_prev + (1 - alpha) * np.sum(grad * grad)
            # x = x_prev + lr * (grad*rho + (1-rho)*grad_prev) / (ri ** 0.5 + 1.e-6)

            # =========================== Method 3
            # rho = 0.1
            # direction = grad / np.linalg.norm(grad)
            # x = x_prev + lr * (direction * (1 - rho) + rho * direction_prev)

            # =====================================================
            direction = grad / np.linalg.norm(grad)
            x = x_prev + direction * lr

            if objective < objective_prev:
                lr *= 0.2
                # print(f"Reduced learning rate to {lr:.3e}")
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
            grad_prev = grad
            direction_prev = direction

            if convergence_steps % 10 == 0:
                print(f"Iteration {convergence_steps} - objective value {objective} - parameter values: {x * np.array([scaling_parameter, 1])} - stepsize {lr}")

    except KeyboardInterrupt:
        converged = True

    return x, convergence_steps, objective, np.array(x_history)


def linesearch(starting_point, direction, objective_func, analysis_func, constraints, tolerance):
    # Stage one: find endpoint of the linesearch
    dir = direction / np.linalg.norm(direction)
    # limits on [a1, midpoint height]
    xmin = np.array([0, -0.175])
    xmax = np.array([5, 0])

    delta_lower = 0
    # For each dimension, calculate delta_upper where the point reaches xmax or xmin
    delta_upper_pos = (xmax - starting_point) / dir
    delta_upper_neg = (xmin - starting_point) / dir

    # Filter out negative deltas to find the max positive delta
    delta_upper = np.min(np.where(dir > 0, delta_upper_pos, delta_upper_neg))

    # Ensure delta_upper is positive
    delta_upper = max(0, delta_upper)

    # Stage two: find optimum between the two
    delta_optimum = golden_section(starting_point, dir, delta_lower, delta_upper, objective_func, analysis_func, constraints, tolerance)
    return starting_point + dir * delta_optimum


def golden_section(starting_point, direction, delta_lower, delta_upper, objective_function, analysis_function, constraints, tolerance):
    phi = (5**0.5 - 1)/2

    probe1_delta = delta_lower + (delta_upper - delta_lower) * (1-phi)
    xprobe1 = starting_point + direction * probe1_delta
    f_probe1 = objective_function(analysis_function, xprobe1, constraints)

    probe2_delta = probe1_delta + (delta_upper - probe1_delta) * (1-phi)
    xprobe2 = starting_point + direction * probe2_delta
    f_probe2 = objective_function(analysis_function, xprobe2, constraints)

    if abs(f_probe1 - f_probe2) < tolerance:
        return probe1_delta

    if f_probe2 > f_probe1:
        # Search between probe1 and upper
        return golden_section(starting_point, direction, probe1_delta, delta_upper, objective_function, analysis_function,
                       constraints, tolerance)
    else:
        return golden_section(starting_point, direction, delta_lower, probe2_delta, objective_function, analysis_function,
                       constraints, tolerance)


def optimize_cg(analysis_func, objective_func, params, constraints, lr, tol, h):
    x = copy.deepcopy(params)
    x_history = [x]

    grad_prev, objective_prev = gradient(f_analysis=analysis_func,
                               f_obj=objective_func,
                               params=x,
                               constraints=constraints,
                               h=h)

    search_dir_prev = grad_prev# / np.linalg.norm(grad_prev)
    # search_dir_prev = grad_prev


    convergence_steps = 0
    converged = False

    try:
        while not converged:
            convergence_steps += 1

            grad, objective = gradient(f_analysis=analysis_func,
                                       f_obj=objective_func,
                                       params=x,
                                       constraints=constraints,
                                       h=h)

            if np.linalg.norm(grad) < tol:
                print(f"\n\nConverged at: \nIteration {convergence_steps} - objective value {objective} - parameter values: {x}")
                converged = True
                # Finish iteration

            if convergence_steps < 2:
                search_dir = search_dir_prev
            else:

                # search_dir = grad - search_dir_prev * (np.linalg.norm(grad) / np.linalg.norm(grad_prev))**2
                search_dir = grad - search_dir_prev * (np.dot(grad,grad) / np.dot(grad_prev,grad_prev))
                # search_dir = grad - search_dir_prev * (
                #             np.linalg.norm(grad) / np.linalg.norm(grad_prev)) ** 2
            #search_dir /= np.linalg.norm(search_dir)

            # Linesearch update
            x = linesearch(x, search_dir, objective_func, analysis_func, constraints, tol)

            x_history.append(copy.deepcopy(x))

            grad_prev = grad
            search_dir_prev = search_dir

            if convergence_steps % 2 == 0:
                print(f"Iteration {convergence_steps} - objective value {objective} - parameter values: {x}")

    except KeyboardInterrupt:
        converged = True

    return x, convergence_steps, objective, np.array(x_history)

def simplified_analysis(*params, **kwargs):
    scaling_parameter = kwargs.get('scaling_parameter', None)
    if scaling_parameter is None:
        scaling_parameter = 1

    pars = [params[0]*scaling_parameter,      # a1
            10,             # N
            12,             # tanwidth
            4,              # radwidth
            params[1]]      # midheight

    return main(*pars)

# def inv_objective(params, analysis_function, constraints):
#     return - objective(analysis_function, params,constraints)


if __name__ == '__main__':
    scaling_parameter = 5
    initial_params = np.array([1.8 / scaling_parameter, -0.02])
    # initial_params = np.array([4 / scaling_parameter, -0.175])
    lr = 5e-2
    tol = 1e-5
    h = 1e-8

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

    logger("log_files", "optimization",
           initial_parameters=str(initial_params * np.array([scaling_parameter, 1])),
           constraints=str(constraints),
           learning_rate=lr,
           tolerance=tol,
           steps=steps,
           optimal_solution=optimal * np.array([scaling_parameter,1]),
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

    # print(objective(simplified_analysis, [2.21, -0.123], constraints))
    # print(opt.minimize(inv_objective, initial_params))
