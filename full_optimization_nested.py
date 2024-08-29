import numpy as np
import copy
from main import main
from logger import logger
from generate_bridge import generate_bridge


def gLmax(L, Lmax, p, p2):
    return p * np.exp((L - Lmax)/(p2 * Lmax))


def gLmin(L, Lmin, p, p2):
    return p / np.exp((L - Lmin)/(p2 * Lmin))


def gMmax(M, Mmax, p, p2):
    return p * np.exp((M - Mmax)/(p2 * Mmax))


def objective(params, N, max_constraints, p1, p2, **kwargs):
    failure_mass, mass_bridge, min_length, max_length = full_analysis(params, N, **kwargs)
    c_min_length, c_max_length, c_max_mass = max_constraints
    penalties = [gMmax(mass_bridge, c_max_mass, p=p1, p2=p2), gLmin(min_length, c_min_length, p=p1, p2=p2), gMmax(max_length, c_max_length, p=p1, p2=p2)]
    obj = failure_mass
    obj -= sum(penalties)
    return obj


def gradient(params, N, constraints, h, p1, p2, **kwargs):
    grad = np.zeros(len(params))
    centre_eval = objective(params, N, constraints, p1, p2, **kwargs)
    for i in range(len(params)):
        modification = np.zeros(params.shape)
        modification[i] += h
        eval = objective(params + modification, N, constraints, p1, p2, **kwargs)
        grad[i] = (eval - centre_eval) / h
    return grad, centre_eval


def golden_section_it(n_vals, obj_vals, x_vals, constraints, tolerance, h_inner, lr_inner, p1, p2, **kwargs):
    lower_n, probed_n, upper_n = n_vals
    lower_obj, probed_obj, upper_obj = obj_vals
    lower_x, probed_x, upper_x = x_vals

    phi = (5 ** 0.5 - 1) / 2

    if abs(probed_n - lower_n) < (upper_n - probed_n):
        probe1_n, probe1_obj, probe1_x = probed_n, probed_obj, probed_x
        probe2_n = int(round(probe1_n + phi * (upper_n - probe1_n)))
        if probe2_n != upper_n:
            probe2_x, convergence_steps, probe2_obj, param_history_probe2, continue_flag = optimize_inner(
                params=upper_x,
                N=probe2_n,
                constraints=constraints,
                lr=lr_inner,
                tol=tolerance,
                h=h_inner,
                p1=p1, p2=p2, **kwargs)
        else:
            probe2_x, convergence_steps, probe2_obj, param_history_probe2, continue_flag = upper_x, 0, upper_obj, upper_x, True

    # The new probe point is the one on the left
    else:
        probe2_n, probe2_obj, probe2_x = probed_n, probed_obj, probed_x
        probe1_n = int(lower_n + phi * (probe2_n - lower_n))

        if probe1_n != lower_n:
            probe1_x, convergence_steps, probe1_obj, param_history_probe1, continue_flag = optimize_inner(
                params=upper_x,
                N=probe1_n,
                constraints=constraints,
                lr=lr_inner,
                tol=tolerance,
                h=h_inner,
                p1=p1, p2=p2, **kwargs)
        else:
            probe1_x, convergence_steps, probe1_obj, param_history_probe1, continue_flag = lower_x, 0, lower_obj, lower_x, True

    # probe1_N = int(lower_val + (upper_val - lower_val) * (1-phi))
    # probe2_N = int(round(probe1_N + (upper_val - probe1_N) * (1 - phi)))
    # print(f'\n N1: {probe1_n}, N2: {probe2_n}')
    obs_max_idx = np.argmax([obj_vals[0], probe1_obj, probe2_obj, obj_vals[-1]])

    if obs_max_idx >= 2:
        # The minimum is either at the right boundary or probe2 -> Search between probe1 and upper
        return (probe1_n, probe2_n, upper_n), (probe1_obj, probe2_obj, upper_obj), (probe1_x, probe2_x, upper_x), convergence_steps, continue_flag
    else:
        # The minimum is either at the left boundary or probe1 -> Search between lower and probe2
        return (lower_n, probe1_n, probe2_n), (lower_obj, probe1_obj, probe2_obj), (lower_x, probe1_x, probe2_x), convergence_steps, continue_flag


def optimize_inner(params, N, constraints, lr, tol, h, p1, p2, **kwargs):
    continue_opt = True
    scaling_pars = kwargs.get('scaling_parameters', None)
    if scaling_pars is None:
        scaling_pars = np.ones_like(params)

    # Initial params are scaled down to prevent double scaling when using a previous (scaled) output as input
    x_prev = copy.deepcopy(params) / scaling_pars
    x = copy.deepcopy(x_prev)
    x_history = [x]

    objective_prev = -1e10
    obj = objective_prev

    convergence_steps = 0
    converged_inner = False

    try:
        while not converged_inner:
            convergence_steps += 1

            grad, obj = gradient(params=x,
                                       N=N,
                                       constraints=constraints,
                                       h=h, p1=p1, p2=p2, **kwargs)

            direction = grad / np.linalg.norm(grad)
            x = x_prev + direction * lr

            if obj < objective_prev:
                lr *= 0.2
            else:
                lr *= 1.02

            if abs((obj - objective_prev) / obj) <= tol:
                print(f"N = {N}: Converged at inner iteration {convergence_steps} - objective value {obj:3f} - parameter values: {x * scaling_pars}")
                converged_inner = True

            if obj < -10**10 and ((obj-objective_prev) / obj) < 0.1:
                print(f"Stopping iteration for N = {N} after {convergence_steps} iterations as the objective value \
                is too low and a valid solution is unlikely to be found - objective value {obj:3f}")
                converged_inner = True

            # if np.linalg.norm(grad) <= tol:
            #     print(f"Iteration {convergence_steps} - objective value {objective} - parameter values: {x}")
            #     converged = True

            x_history.append(copy.deepcopy(x))
            objective_prev = obj
            x_prev = x

            if convergence_steps % 10 == 0:
                avg_gradient_tracker.append((N, grad))

            if convergence_steps % 50 == 0:
                print(f"Inner iteration {convergence_steps} for N={N} - objective value {obj:.2f} - parameter values: {np.round(x * scaling_pars,3)} - stepsize {lr:.2e}")

    except KeyboardInterrupt:
        print(f"\n\nKeyboard interrupt detected in inner loop - Exiting\n")
        converged_inner = True
        continue_opt = False    # Set to false to exit the outer optimization

    # Compare the rounded discrete vars
    # Before rounding, set the variable x to its actual scaled value that we have been optimising for
    # Otherwise rounding is very rough
    # As a consequence DO NOT USE SCALING PARAMETER ON THIS X AFTER THIS!
    # THEREFORE DO NOT PASS **KWARGS!
    x_true = x * scaling_pars
    r_up_up     = np.array([x_true[0], np.ceil(x_true[1]), np.ceil(x_true[2]), x_true[3]])
    r_up_down   = np.array([x_true[0], np.ceil(x_true[1]), np.floor(x_true[2]), x_true[3]])
    r_down_up   = np.array([x_true[0], np.floor(x_true[1]), np.ceil(x_true[2]), x_true[3]])
    r_down_down = np.array([x_true[0], np.floor(x_true[1]), np.floor(x_true[2]), x_true[3]])

    x_variations = [r_up_up, r_up_down, r_down_up, r_down_down]
    # print(x_variations)
    obj_variations = [objective(x_var, N, constraints, p1, p2) for x_var in x_variations]
    print('Values for rounded x: ', obj_variations)
    print('values for float x: ', objective(x, N, constraints, p1, p2, **kwargs))
    max_idx = np.argmax(obj_variations)
    print(f'Final objective function for N = {N}: {obj_variations[max_idx]:.2f} for x = {np.round(x_variations[max_idx], 3)}')
    return x_variations[max_idx], convergence_steps, obj_variations[max_idx], np.array(x_history), continue_opt


def optimize(inner_params, n_range, constraints, lr, tol, h, p1=100, p2=0.005, **kwargs):
    optimal_x_history = []

    # initialisation
    lower_n, upper_n = n_range
    probe_n = int(lower_n + ((5 ** 0.5 - 1) / 2) * (upper_n - lower_n))
    lower_n_x, lower_n_its, lower_n_obj, lower_n_history, continue_opt = optimize_inner(params=inner_params, N=lower_n,
                                                                                        constraints=constraints, lr=lr,
                                                                                        tol=tol, h=h, p1=p1, p2=p2, **kwargs)
    probe_n_x, probe_n_its, probe_n_obj, probe_n_history, continue_opt = optimize_inner(params=inner_params, N=probe_n,
                                                                                        constraints=constraints, lr=lr,
                                                                                        tol=tol, h=h, p1=p1, p2=p2, **kwargs)
    upper_n_x, upper_n_its, upper_n_obj, upper_n_history, continue_opt = optimize_inner(params=inner_params, N=upper_n,
                                                                                        constraints=constraints, lr=lr,
                                                                                        tol=tol, h=h, p1=p1, p2=p2, **kwargs)
    n_vals, obj_vals, x_vals = (lower_n, probe_n, upper_n), (lower_n_obj, probe_n_obj, upper_n_obj), (lower_n_x, probe_n_x, upper_n_x)

    outer_iterations = 0
    total_steps = lower_n_its + upper_n_its + probe_n_its
    try:
        converged = False
        while not converged:
            outer_iterations += 1
            print(
                f"\n\nOuter iteration {outer_iterations} - n range:{n_vals} \nObjective values: {obj_vals} \nTotal steps: {total_steps}\n")

            n_vals, obj_vals, x_vals, iterations, continue_opt = golden_section_it(
                n_vals=n_vals, obj_vals=obj_vals, x_vals=x_vals,
                constraints=constraints,
                tolerance=tol, h_inner=h,
                lr_inner=lr, p1=p1, p2=p2, **kwargs)

            optimal_x_history.append((x_vals))
            total_steps += iterations

            if not continue_opt or (abs((obj_vals[2] - obj_vals[0]) / obj_vals[0]) < tol) or (n_vals[2] - n_vals[0] <= 2):
                converged = True
                print(f'Final n values: {n_vals} with objective values: {obj_vals}')

    except KeyboardInterrupt:
        converged = True

    max_obj_idx = obj_vals.index(max(obj_vals))
    opt_params = x_vals[max_obj_idx]
    opt_n = n_vals[max_obj_idx]
    opt_obj = obj_vals[max_obj_idx]

    return opt_params, opt_n, opt_obj, optimal_x_history, total_steps, outer_iterations


def full_analysis(params, N, **kwargs):
    scaling_parameters = kwargs.get('scaling_parameters', None)
    if scaling_parameters is None:
        scaling_parameters = np.ones_like(params)
        # print(f'Full analysis call {scaling_parameters * params}')

    pars_scaled = scaling_parameters * params

    pars = [pars_scaled[0],     # a1
            N,                  # N
            pars_scaled[1],     # tanwidth
            pars_scaled[2],     # radwidth
            pars_scaled[3]]     # midheight

    return main(*pars)


if __name__ == '__main__':

    ## x = a, tanwidth, radwidth, midheight
    # scaling_parameters = np.array([6,10,5,1])
    # scaling_parameters = np.array([5, 100, 10, 1])
    scaling_parameters = np.array([5, 30, 40, 1])

    initial_params = np.array([1.8, 12, 4, -0.02])
    N_range = [2, 20]

    lr = 5e-2
    tol = 1e-9
    h = 1e-8

    # Penalty function parameters
    p1 = 10
    p2 = 0.009

    # Min length, max length, max mass
    constraints = [0.03, 0.3, 0.5]

    avg_gradient_tracker = []

    optimal_params, optimal_N, objective_val, x_history, total_its, outer_its = optimize(inner_params=initial_params,
                                                                                         n_range=N_range,
                                                                                         constraints=constraints,
                                                                                         lr=lr, tol=tol, h=h,
                                                                                         scaling_parameters=scaling_parameters,
                                                                                         p1=p1, p2=p2)

    carried_mass, bridge_mass, min_length, max_length = full_analysis(optimal_params, optimal_N)

    print(f"Final objective value: {objective_val} after {total_its} iterations with N = {optimal_N} and params = {optimal_params}")
    boundaries_voilated = not (min_length > constraints[0] and max_length < constraints[1] and bridge_mass < constraints[2])
    logger("log_files_full", "optimization",
           initial_parameters=str(initial_params),  # * scaling_parameters),
           constraints=str(constraints),
           learning_rate=lr,
           tolerance=tol,
           total_inner_its=total_its,
           outer_its=outer_its,
           optimal_inner=optimal_params,  # * scaling_parameters,
           optimal_N=optimal_N,
           objective_value=objective_val,
           carried_mass=carried_mass,
           bridge_mass=bridge_mass,
           minimum_spaghetti_length=min_length,
           maximum_spaghetti_length=max_length,
           constraints_violated=boundaries_voilated,
           p1=p1,
           p2=p2,
           scaling_pars=scaling_parameters
           )

    opt_par_scaled = optimal_params #* scaling_parameters
    generate_bridge(a1=opt_par_scaled[0],
                    N=optimal_N,
                    tanWidth=opt_par_scaled[1],
                    radWidth=opt_par_scaled[2],
                    mid_height=opt_par_scaled[3],
                    plotting=True)
