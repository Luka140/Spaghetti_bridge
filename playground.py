import itertools
import pandas as pd
from main import main
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# -0.16,20,6,6,760.8294506862147,0.4285679568355713,0.2968164415931166
# 2     -0.12        18         4        22  1426.366642       0.494732  0.277308

# 2.0     -0.14        17         6        10  1371.255823       0.489268  0.286531  0.031159
# 2.0     -0.14        18         6         9  1392.098887       0.490239  0.286531  0.035049
# 2.25     -0.13        17         6        10  1438.768245       0.496822  0.28178  0.032289
# 2.2    -0.135        17         6        10  1448.510615       0.497663  0.284121  0.032116

# Continuous variables
parabola_slope = [2.18, 2.2, 2.22]
midpoint = [ -0.1379, -0.1377, -0.1375]

# Discrete variables
tanwidth = [16, 17, 18]
radwidth = [5, 6, 7]
num_arcs = [9, 10, 11]

levels = [parabola_slope, midpoint, tanwidth, radwidth, num_arcs]

def getDOEmatrix(levels):
    design_matrix = list(itertools.product(*levels))
    columns = ['parabola_slope', 'midpoint', 'tanwidth', 'radwidth', 'num_arcs']

    df_design = pd.DataFrame(design_matrix, columns=columns)

    df_design.to_csv('bridge_factorial_design.csv', index=False)


# main(a1, n, tanwidth, radwidth, mid_h, plotting=False)
def run(row):
    a1 = row['parabola_slope']
    midpoint = row['midpoint']
    tanwidth = row['tanwidth']
    radwidth = row['radwidth']
    n = row['num_arcs']

    print(a1, n, tanwidth, radwidth, midpoint)
    return main(a1, int(n), tanwidth, radwidth, midpoint, plotting=False)


getDOEmatrix(levels)

matrix = pd.read_csv('bridge_factorial_design.csv')

max_mass_results = []
bridge_weight_results = []
max_L = []
min_L =[]

for index, row in matrix.iterrows():
    #failure_mass, np.max(lengths), mass_bridge, nodeconstraint
    max_mass, L, bridge_weight, Lmin= run(row)
    max_mass_results.append(max_mass)
    bridge_weight_results.append(bridge_weight)
    max_L.append(L)
    min_L.append(Lmin)

try:
    matrix['max_mass'] = max_mass_results
    matrix['bridge_weight'] = bridge_weight_results
    matrix['max_L'] = max_L
    matrix['min_L'] = min_L

    matrix.to_csv('bridge_factorial_design_with_responses.csv', index=False)
    print("File saved successfully.")
except Exception as e:
    print(f"Failed to save the file: {e}")

MAX_BRIDGE_MASS = 0.5
MAX_NOODLE_LENGTH = 0.3
MIN_NOODLE_LENGTH = 0.03

filtered = matrix[(matrix['bridge_weight'] <= MAX_BRIDGE_MASS) & (matrix['max_L'] <= MAX_NOODLE_LENGTH) & (matrix['min_L'] >= MIN_NOODLE_LENGTH) ]
sorted_df = filtered.sort_values(by='max_mass', ascending=False)
sorted_df.to_csv('sorted_results.csv', index=False)
print(sorted_df.head(1))

# Calculate the correlation matrix
correlation_matrix = matrix[['parabola_slope', 'midpoint', 'tanwidth', 'radwidth', 'num_arcs', 'max_mass']].corr()

# Extract correlations with the target variable 'max_mass'
correlations_with_target = correlation_matrix['max_mass'].drop('max_mass')

# Print the correlations
print("Correlation with max_mass:")
print(correlations_with_target)


X = matrix[['parabola_slope', 'midpoint', 'tanwidth', 'radwidth', 'num_arcs']]
y = matrix['max_mass']

# Initialize and fit the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a Partial Dependence Plot
features = ['parabola_slope', 'midpoint', 'tanwidth', 'radwidth', 'num_arcs']
fig, ax = plt.subplots(figsize=(12, 8))

# Plot PDPs
disp = PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
plt.suptitle('Partial Dependence Plots')
plt.subplots_adjust(top=0.9)  # Adjust the title position
plt.show()