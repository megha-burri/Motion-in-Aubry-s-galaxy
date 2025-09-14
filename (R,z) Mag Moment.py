import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter


# Constants
M = 1.0  
E = 0.6
L_z = 6
D = 10

# Gamma 
def gamma(R, z, p_R, p_z, L_z):
    r = np.sqrt(R**2 + z**2)
    return np.sqrt(1 + p_R**2 + p_z**2 + (L_z / R - D * R / r**3)**2)

# Equations of Motion
def system(t, state):
    R, z, p_R, p_z = state
    r = np.sqrt(R**2 + z**2)
    gamma_val = gamma(R, z, p_R, p_z, L_z)

    dR_dt = p_R / gamma_val
    dz_dt = p_z / gamma_val

    dp_R_dt = -((1 / gamma_val)
        * (L_z / R - D * R / r**3)
        * (-L_z / R**2 - D / r**3 + 3 * D * R**2 / r**5)
        + M * R / r**3)

    dp_z_dt = -((1 / gamma_val) * (L_z / R - D * R / r**3)
                * (3 * D * R * z / r**5) + M * z / r**3)

    return [dR_dt, dz_dt, dp_R_dt, dp_z_dt]


def compute_initial_pz(R_0, z_0, p_R_0):
    r_0 = np.sqrt(R_0**2 + z_0**2)
    return np.sqrt((E + 1 / r_0)**2 - 1 - (L_z / R_0 - D * R_0 / r_0**3)**2 - p_R_0**2)


R0, z0, pR0 = 1.5, 0.0, 0.0
pz0 = compute_initial_pz(R0, z0, pR0)
y0 = [R0, z0, pR0, pz0]
t_span = (0, 75)
sol = solve_ivp(system, t_span, y0, method='LSODA', dense_output=True, rtol=1e-10, atol=1e-12)
print(pz0)

t=sol.t
R, z, p_R, p_z = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
r = np.sqrt(R**2 + z**2)
p_phi = (L_z/R) - (D*R)/r**3

#Magnetic Moment
p_sqr = p_R**2 + p_phi**2 + p_z**2

B_R = 3*z*R*D/r**5
B_z = D*((3*z**2/r**5) - 1/r**3)
B_mag = (D/r**4)*np.sqrt(R**2 + 4*z**2)
p_par = (p_R*B_R + p_z*B_z)/B_mag

p_perpsq = p_sqr - p_par**2

mu = p_perpsq/B_mag



# Plot Trajectory
plt.figure(figsize=(8, 6), dpi = 600)
plt.plot(R, z, color='blue')

plt.xlabel("R", fontsize=25)
plt.ylabel("z", fontsize=25)
plt.tick_params(axis='both', labelsize=20)

plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Define Hill's region
grid_size = 2000
R_vals = np.linspace(0.0001, 2, grid_size)
z_vals = np.linspace(-1, 1, grid_size)
R_grid, Z_grid = np.meshgrid(R_vals, z_vals)
r_grid = np.sqrt(R_grid**2 + Z_grid**2)
expression = (E + 1 / r_grid)**2 - 1 - (L_z / R_grid - D * R_grid / r_grid**3)**2

# Plot Hill's region
plt.contour(R_grid, Z_grid, expression, levels=[0], colors='red', linewidths = 3.5)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.grid(True, linestyle = '--')
plt.show()

# Plot Magnetic Moment
plt.figure(figsize=(8, 6), dpi=600)
plt.plot(t, mu, color='crimson')
plt.xlabel('t', fontsize= 25)
plt.ylabel(r'$\mu$', fontsize=25)
plt.tick_params(axis='both', labelsize=20)

plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.grid(True)
plt.tight_layout()
plt.show()
