import numpy as np 
import copy
from main import main
from logger import logger

def gLmax(L, Lmax, p=100):
    return 100 * np.exp(L - Lmax)**p

def gLmin(L, Lmin, p=100):
    return 100 / np.exp(L - Lmin)**p

def gMmax(M, Mmax, p=100):
    return 100 * np.exp(M - Mmax)**p


def gMmin(M, Mmin, p=100):
    return 100 / np.exp(M - Mmin)**p


def objective(analysis_function, params, max_constraints):
    failure_mass, mass_bridge, min_length, max_length = analysis_function(params)
    c_min_length, c_max_length, c_max_mass = max_constraints
    penalties = [gMmax(mass_bridge, c_max_mass), gLmin(min_length, c_min_length), gMmax(max_length, c_max_length)]
    print(f"Carrying {failure_mass:2f} kg - bridge mass: {mass_bridge:2f} kg - length range {min_length:2f} {max_length:2f}")
    obj = failure_mass
    obj -= sum(penalties)
    return obj


def gradient(f_analysis, f_obj, params, constraints):
    h = 1e-5
    grad = np.zeros(len(params))
    centre_eval = f_obj(f_analysis, params, constraints)
    for i in range(len(params)):
        modification = np.zeros(params.shape)
        modification[i] += h
        eval = f_obj(f_analysis, params + modification, constraints)
        grad[i] = (eval - centre_eval) / h
    return grad, centre_eval


# def steespest_descent(F, params, lr, tol):
#     grad = gradient(F, params)
#     while np.linalg.norm(grad) > tol:
#         params -= lr * grad
#         grad = gradient(F, params)
#     return params

# print(steespest_descent(F, [1, 0.25], 1e-3, 1e-6))

def optimize(analysis_func, objective_func, params, constraints, lr, tol):
    x_prev = copy.deepcopy(params)
    x = copy.deepcopy(params)
    objective_prev = 0

    convergence_steps = 0
    converged = False
    while not converged:
        convergence_steps += 1
        grad, objective = gradient(f_analysis=analysis_func,
                                   f_obj=objective_func,
                                   params=x,
                                   constraints=constraints)
        # print(grad, x_prev)
        x = x_prev + lr * grad / np.linalg.norm(grad)
        # print(x, objective)
        if abs(objective - objective_prev) <= tol:
            converged = True

        x_prev = x
        # print(x, objective)

    return x, convergence_steps, objective


def simplified_analysis(params):
    pars = {'a1':params[0],
            'n':20,
            'tanwidth':12,
            'radwidth':4,
            'mid_h': params[1]
    }
    return main(**pars)


if __name__ == '__main__':
    initial_params = np.array([1.8, 0])
    lr = 1e-2
    tol = 1e-6

    # Min length, max length, max mass
    constraints = [0.03, 0.25, 0.3]
    optimal, steps, objective_val = optimize(simplified_analysis, objective, initial_params, constraints, lr, tol)

    carried_mass, bridge_mass, min_length, max_length = simplified_analysis(optimal)

    logger("log_files", "optimization",
           initial_parameters=str(initial_params),
           constraints=str(constraints),
           learning_rate=lr,
           tolerance=tol,
           steps=steps,
           optimal_solution=optimal,
           objective_value=objective_val,
           carried_mass=carried_mass,
           bridge_mass=bridge_mass,
           minimum_spaghetti_length=min_length,
           maximum_spaghetti_length=max_length
           )
