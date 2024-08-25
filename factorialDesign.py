import itertools
import pandas as pd
from main import main
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# -0.16,20,6,6,760.8294506862147,0.4285679568355713,0.2968164415931166

# # Continuous variables
# parabola_slope = [1, 4, 6]
# midpoint = [-0.16, -0.08, 0.0]

# # Discrete variables
# Na_arc = [4, 12, 22]
# Na_rad = [4, 12, 18]
# N_col = [6, 12, 18]

# Continuous variables
parabola_slope = [1, 4, 6]
midpoint = [-0.16, 0, 0.16]

# Discrete variables
Na_arc = [2, 10, 20]
Na_rad = [2, 10, 20]
N_col = [2, 10, 20]

levels = [parabola_slope, midpoint, Na_arc, Na_rad, N_col]

def getDOEmatrix(levels):
    design_matrix = list(itertools.product(*levels))
    columns = ['parabola_slope', 'midpoint', 'Na_arc', 'Na_rad', 'N_col']

    df_design = pd.DataFrame(design_matrix, columns=columns)

    df_design.to_csv('bridge_factorial_design.csv', index=False)


# main(a1, n, Na_arc, Na_rad, mid_h, plotting=False)
def run(row):
    a1 = row['parabola_slope']
    midpoint = row['midpoint']
    Na_arc = row['Na_arc']
    Na_rad = row['Na_rad']
    n = row['N_col']

    print(a1, n, Na_arc, Na_rad, midpoint)
    return main(a1, int(n), Na_arc, Na_rad, midpoint, plotting=False)


getDOEmatrix(levels)

matrix = pd.read_csv('bridge_factorial_design.csv')

max_mass_results = []
bridge_weight_results = []
max_L = []

for index, row in matrix.iterrows():
    #failure_mass, np.max(lengths), mass_bridge, nodeconstraint
    max_mass, L, bridge_weight, _ = run(row)
    max_mass_results.append(max_mass)
    bridge_weight_results.append(bridge_weight)
    max_L.append(L)

try:
    matrix['max_mass'] = max_mass_results
    matrix['bridge_weight'] = bridge_weight_results
    matrix['max_L'] = max_L

    matrix.to_csv('bridge_factorial_design_with_responses.csv', index=False)
    print("File saved successfully.")
except Exception as e:
    print(f"Failed to save the file: {e}")

MAX_BRIDGE_MASS = 0.5
MAX_NOODLE_LENGTH = 0.3

filtered = matrix[(matrix['bridge_weight'] <= MAX_BRIDGE_MASS) & (matrix['max_L'] <= MAX_NOODLE_LENGTH)]
sorted_df = filtered.sort_values(by='max_mass', ascending=False)
sorted_df.to_csv('sorted_results.csv', index=False)
print(sorted_df.head(5))

# Calculate the correlation matrix
correlation_matrix = matrix[['parabola_slope', 'midpoint', 'Na_arc', 'Na_rad', 'N_col', 'max_mass']].corr()

# Extract correlations with the target variable 'max_mass'
correlations_with_target = correlation_matrix['max_mass'].drop('max_mass')

# Print the correlations
print("Correlation with max_mass:")
print(correlations_with_target)


X = matrix[['parabola_slope', 'midpoint', 'Na_arc', 'Na_rad', 'N_col']]
y = matrix['max_mass']

# Initialize and fit the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a Partial Dependence Plot
features = ['parabola_slope', 'midpoint', 'Na_arc', 'Na_rad', 'N_col']
fig, ax = plt.subplots(figsize=(12, 8))

# Plot PDPs
disp = PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
plt.suptitle('Partial Dependence Plots')
plt.subplots_adjust(top=0.9)  # Adjust the title position
plt.show()