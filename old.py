import numpy as np
import matplotlib.pyplot as plt
import HW3FrameStiffnessMatrix as HW3f


# boundary conditions; form: [degrees of freedom], [applied displacements]
BC = np.array([[1, 2, 3], [0, 0, 0]])

E = 209e3  # E-modulus
width, height, length = 70, 25, 1000
# distributed transverse load format: ax + b (use positive numbers for a downwards transverse load)
a = 0.001
b = 5.
# distributed constant axial load: qx N/mm (positive pulls towards positive x)
qx = 1.5

# The number of elements to be used if only a single simulation is run (set convergence_check to False)
n_elements = 2

# The maximum element count to be considered for the convergence test (set convergence_check to True)
max_elements = 50
convergence_check = True


if __name__ == '__main__':

    if convergence_check:
        print("Performing convergence analysis")
        HW3f.check_convergence(E, range(1, max_elements + 1), a, b, qx, width, height, length, BC)
        print("Finished")
    else:
        u, reactions = HW3f.simulate_frame(E, n_elements, a, b, qx,width,height,length, BC)
        u_res = np.reshape(u, (-1, 3))
        plt.plot(np.linspace(0, length, u_res.shape[0]), u_res[:,1])
        plt.show()
        for i in range(n_elements + 1):
            print(f"Node {i+1} axial displacement: {u_res[i,0]}")
            print(f"Node {i + 1} transverse displacement: {u_res[i, 1]}")
            print(f"Node {i + 1} rotation: {u_res[i, 2]} \n")

        print("Reaction forces:")
        non_zero_reactions = [(DoF, reactions[DoF]) for DoF in range(reactions.size) if abs(reactions[DoF]) > np.max(abs(reactions))/10**6]
        for i in range(len(non_zero_reactions)):
            print(f"Node {non_zero_reactions[i][0]//3 + 1} DoF {non_zero_reactions[i][0] % 3}: {non_zero_reactions[i][1]}")