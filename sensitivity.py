import numpy as np 
from main import main
import matplotlib.pyplot as plt

def gLmax(L, Lmax, p=20):
    if L>=Lmax:
        print('L>=Lmax', L)
    return 100*np.exp(L - Lmax)**p 

def gLmin(L, Lmin, p=20):

    return 1/np.exp(L - Lmin)**p 

def gMmax(M, Mmax, p=20):
    return np.exp(M - Mmax)**p 

def gMmin(M, Mmin, p=20):
    return 1/np.exp(M - Mmin)**p 

Lmax = 0.3
Lmin = 0.03
Mmax = 0.5*0.9
Mmin = 0

# sensitivity analysis
def objective_function(a1, n, tanwidth, radwidth, mid_h, returnL=False):
    failure_mass, L, mass, _ = main(a1, n, tanwidth, radwidth, mid_h)
    if returnL:
        return L
    return failure_mass - gLmax(L,Lmax) - gLmin(L, Lmin) - gMmax(mass, Mmax) -gMmin(mass, Mmin)

N = 20
A1 = np.linspace(1,6,N)
MID = np.linspace(-0.16, 0.16, N)

a_grid, mid_grid = np.meshgrid(A1, MID, indexing='ij')



a1_sens = np.zeros((N,N))
mid_sens = np.zeros((N,N))

#sensitivity analysis
# Choose optimum h 
hmid = np.logspace(-16, 1, 40) 
h = np.logspace(-16, 1, 40) 
hrad = np.logspace(-16, 1, 40)
htan = np.logspace(-16, 1, 40)


print(hmid)
grad_alpha = []
grad_mid = []
grad_rad = []
grad_tan = []
# noise
for i in range(len(h)-1):
    grad_alpha.append(np.abs((main(2+h[i+1], 20, 12, 4, -0.07)[0] - main(2-h[i+1], 20, 12, 4, -0.07)[0])/(2*h[i+1])))
    grad_mid.append(np.abs((main(2, 20, 12, 4, -0.07+hmid[i+1])[0] - main(2, 20, 12, 4, -0.07-hmid[i+1])[0])/(2*hmid[i+1])))
    grad_rad.append(np.abs((main(2, 20, 12+hrad[i+1], 4, -0.07)[0] - main(2, 20, 12-hrad[i+1], 4, -0.07)[0])/(2*hrad[i+1])))
    grad_tan.append(np.abs((main(2, 20, 12, 4+htan[i+1], -0.07)[0] - main(2, 20, 12, 4-htan[i+1], -0.07)[0])/(2*htan[i+1])))

# plot for subplots with labels
plt.close()
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs[0, 0].semilogx(h[1:], grad_alpha)
axs[0, 0].set_title(r'$a$')
axs[0, 1].semilogx(hmid[1:], grad_mid)
axs[0, 1].set_title(r'$h_c$')
axs[1, 0].semilogx(hrad[1:], grad_rad)
axs[1, 0].set_title(r'$N_{A_{rad}}$')
axs[1, 1].semilogx(htan[1:], grad_tan)
axs[1, 1].set_title(r'$N_{A_{col}}$')
plt.show()
# plt.close()
# plt.semilogx(h[1:], grad_alpha, label='parabola slope')
# plt.grid()
# plt.legend()
# # plt.show()
# plt.savefig('noise_alpha.png')

# plt.close()
# plt.semilogx(hmid[1:], grad_mid, label='mid point')
# plt.grid()
# plt.legend()
# plt.savefig('noise_mid.png')



# delta_A1 = 0.4
# delta_mid = 0.02
# #compute A1 sensitivity


# for i in range(N):
#     for j in range(N):
#         a1_sens[i,j] = main(a_grid[i,j] + delta_A1, 20, 12, 4, mid_grid[i,j])[0] - main(a_grid[i,j], 20, 12, 4, mid_grid[i,j])[0]
#         mid_sens[i,j] = main(a_grid[i,j], 20, 12, 4, mid_grid[i,j] + delta_mid)[0] - main(a_grid[i,j], 20, 12, 4, mid_grid[i,j])[0]
# #         a1_sens[i,j] = (objective_function(a_grid[i,j] + delta_A1, 20, 12, 4, mid_grid[i,j]) - objective_function(a_grid[i,j] - delta_A1, 20, 12, 4, mid_grid[i,j]))/(2*delta_A1)
# #         mid_sens[i,j] = (objective_function(a_grid[i,j], 20, 12, 4, mid_grid[i,j] + delta_mid) - objective_function(a_grid[i,j], 20, 12, 4, mid_grid[i,j] - delta_mid))/(2*delta_mid)
# # plt.close()
# # plot 2 3D subplots
# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# ax.plot_surface(a_grid, mid_grid, a1_sens, cmap='viridis')
# ax.set_xlabel('parabola slope')
# ax.set_ylabel('mid point')
# ax.set_zlabel('slope sensitivity')
# ax = fig.add_subplot(122, projection='3d')
# ax.plot_surface(a_grid, mid_grid, mid_sens, cmap='viridis')
# ax.set_xlabel('parabola slope')
# ax.set_ylabel('mid point')
# ax.set_zlabel('mid point sensitivity')
# plt.show()

# convexity = np.zeros((N,N))
# # convexity analysis
# for i in range(N):
#     for j in range(N):
#         convexity[i,j] = main(a_grid[i,j], 20, 12, 4, mid_grid[i,j])[0]

# # plot 3D plot of convexity
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(a_grid, mid_grid, convexity, cmap='viridis')
# ax.set_xlabel('parabola slope')
# ax.set_ylabel('mid point')
# ax.set_zlabel('Max carried mass')
# plt.show()


# plt.close()
# plt.pcolormesh(a_grid, mid_grid, convexity, cmap='viridis')
# plt.colorbar()
# plt.show()

# # plot 2 3D subplots
# # plot 1 3D plot of a1 sensitivity without subplots 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(a_grid, mid_grid, a1_sens, cmap='viridis')
# ax.set_xlabel('parabola slope')
# ax.set_ylabel('mid point')
# ax.set_zlabel('slope sensitivity')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(a_grid, mid_grid, mid_sens, cmap='viridis')
# ax.set_xlabel('parabola slope')
# ax.set_ylabel('mid point')
# ax.set_zlabel('mid point sensitivity')
# plt.show()


# fig = plt.figure()
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')
# ax1.plot_surface(a_grid, mid_grid, np.abs(a1_sens), cmap='plasma')
# ax2.plot_surface(a_grid, mid_grid, np.abs(mid_sens), cmap='plasma')
# ax1.set_xlabel('parabola slope')
# ax1.set_ylabel('mid point')
# ax1.set_zlabel('slope sensitivity')
# ax2.set_xlabel('parabola slope')
# ax2.set_ylabel('mid point')
# ax2.set_zlabel('mid point sensitivity')
# plt.savefig('sensitivity.png')
# obj_mid = []
# obj_a1 = []
# for m in MID:
#     obj_mid.append(objective_function(2, 20, 12, 4, m))

# for a in A1:
#     obj_a1.append(objective_function(a, 20, 12, 4, -0.07))

# plt.close()
# plt.plot(MID, obj_mid, label='mid point')
# plt.legend()
# plt.grid()
# plt.savefig('monot_midpoint.png')
# plt.close()

# plt.plot(A1, obj_a1, label='parabola slope')
# plt.grid()
# plt.legend()
# plt.savefig('monot_slope.png')


# feasibility region 
# feasibility_map = np.zeros((N,N))
# for i in range(N):
#     for j in range(N):
#         flocalL = objective_function(a_grid[i,j], 20, 12, 4, mid_grid[i,j], returnL=True)
#         if flocalL > Lmax or flocalL < Lmin:
#             feasibility_map[i,j] = 1
#         else:
#             feasibility_map[i,j] = 0
# plt.close()
# plt.pcolormesh(a_grid, mid_grid, feasibility_map, cmap='viridis')
# plt.colorbar()
# plt.show()
