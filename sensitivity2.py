from main import main 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

# default main settings 
a1 = 2
n = 20
tanwidth = 12
radwidth = 4
mid_h = -0.07

default_result = main(a1, n, tanwidth, radwidth, mid_h)[0]

# we will now vary the parameters a1, n, tanwidth, radwidth, mid_h by 10 percent
# we will vary the parameters a1, n, tanwidth, radwidth, mid_h by 10 percent
a1_lst = np.linspace(1.8, 2.2, 3)
n_lst = np.array([18, 20, 22])
tanwidth_lst = np.array([11, 12, 13])
radwidth_lst = np.array([3, 4, 5])
mid_h_lst = np.linspace(-0.077, -0.063, 3)

# Prepare to store results
results_a1 = []
results_n = []
results_tanwidth = []
results_radwidth = []
results_mid_h = []

mass_a1 = []
mass_n = []
mass_tanwidth = []
mass_radwidth = []
mass_mid_h = []

# max mass, max L

# Perform sensitivity analysis
for a in a1_lst:
    # Evaluate the function
    result, mass = main(a, n, tanwidth, radwidth, mid_h)[0]
    # Store the results
    results_a1.append(result- default_result)

for n_ in n_lst:
    # Evaluate the function
    result = main(a1, n_, tanwidth, radwidth, mid_h)[0]
    # Store the results
    results_n.append(result- default_result)

for t in tanwidth_lst: 
    # Evaluate the function
    result = main(a1, n, t, radwidth, mid_h)[0]
    # Store the results
    results_tanwidth.append(result- default_result)

for r in radwidth_lst:
    # Evaluate the function
    result = main(a1, n, tanwidth, r, mid_h)[0]
    # Store the results
    results_radwidth.append(result- default_result)

for m in mid_h_lst:
    # Evaluate the function
    result = main(a1, n, tanwidth, radwidth, m)[0]
    # Store the results
    results_mid_h.append(result- default_result)

#plot the results using 5 subplots on 1 line 
fig, axs = plt.subplots(1, 5, figsize=(15, 10))
axs[0].plot(a1_lst, results_a1)
axs[0].set_title('a1 sensitivity')
axs[1].plot(n_lst, results_n)
axs[1].set_title('n sensitivity')
axs[2].plot(tanwidth_lst, results_tanwidth)
axs[2].set_title('tanwidth sensitivity')
axs[3].plot(radwidth_lst, results_radwidth)
axs[3].set_title('radwidth sensitivity')
axs[4].plot(mid_h_lst, results_mid_h)
axs[4].set_title('mid_h sensitivity')

plt.show()

# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# axs[0, 0].plot(a1_lst, results_a1)
# axs[0, 0].set_title('a1 sensitivity')
# axs[0, 1].plot(n_lst, results_n)
# axs[0, 1].set_title('n sensitivity')
# axs[0, 2].plot(tanwidth_lst, results_tanwidth)
# axs[0, 2].set_title('tanwidth sensitivity')
# axs[1, 0].plot(radwidth_lst, results_radwidth)
# axs[1, 0].set_title('radwidth sensitivity')
# axs[1, 1].plot(mid_h_lst, results_mid_h)
# axs[1, 1].set_title('mid_h sensitivity')

# plt.show()

# # Define the range for discrete variables
# tan = [2, 10, 15]  # Example values
# rad = [2, 10, 15]    # Example values
# n = [2, 10, 20]     # Example values

# # Define the range for continuous variables
# a1 = np.linspace(1, 6, 5)  # 5 evenly spaced points between 0 and 10
# midpoint = np.linspace(-0.16, 0.16, 5)   # 5 evenly spaced points between 0 and 5

# # Prepare to store results
# results = []

# # Perform sensitivity analysis
# for t in tan:
#     for r in rad:
#         for n_ in n:
#             for a in a1:
#                 for m in midpoint:
#                     # Evaluate the function
#                     result = main(a, n_, t, r, m)[0]
#                     # Store the results
#                     results.append({
#                         'tan': t,
#                         'rad': r,
#                         'n': n_,
#                         'a1': a,
#                         'midpoint': m,
#                         'result': result
#                     })
# # for d1 in discrete1_values:
# #     for d2 in discrete2_values:
# #         for d3 in discrete3_values:
# #             for c1 in continuous1_range:
# #                 for c2 in continuous2_range:
# #                     # Evaluate the function
# #                     result = f(d1, d2, d3, c1, c2)
# #                     # Store the results
# #                     results.append({
# #                         'd1': d1,
# #                         'd2': d2,
# #                         'd3': d3,
# #                         'c1': c1,
# #                         'c2': c2,
# #                         'result': result
# #                     })

# # Convert results to a structured format if needed, e.g., a DataFrame
# import pandas as pd

# df = pd.DataFrame(results)
# # Save the results to a file
# df.to_csv('results.csv', index=False)


