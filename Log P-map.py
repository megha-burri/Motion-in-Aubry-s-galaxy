import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Constants
M = 1.0  
E = 0.56
L_z = -0.348
D = 1.0

def gamma(R, z, p_R, p_z, L_z):
    r = np.sqrt(R**2 + z**2)
    return np.sqrt(1 + p_R**2 + p_z**2 + (L_z / R - D * R / r**3)**2)

def system(t, state):
    R, z, p_R, p_z = state
    r = np.sqrt(R**2 + z**2)
    gamma_val = gamma(R, z, p_R, p_z, L_z)
    
    dR_dt = p_R / gamma_val
    dz_dt = p_z / gamma_val
    dp_R_dt = -( (1 / gamma_val) * (L_z / R - D * R / r**3)
                 * (-L_z / R**2 - D / r**3 + 3 * D * R**2 / r**5)
                 + M * R / r**3)
    dp_z_dt = -((1 / gamma_val) * (L_z / R - D * R / r**3) * (3 * D * R * z / r**5) + M * z / r**3)
    
    return [dR_dt, dz_dt, dp_R_dt, dp_z_dt]

def poincare_section(t, state):
    return state[1]  # z = 0 plane

poincare_section.terminal = False
poincare_section.direction = 1  # p_z>0

# Initial conditions
z_0 = 0.0
R_0 = 1.5
p_R_0 = 0.0
r_0 = np.sqrt(R_0**2 + z_0**2)
p_z_0 = np.sqrt((E + 1/r_0)**2 - 1 - (L_z / R_0 - D * R_0 / r_0**3)**2 - p_R_0**2)
print(p_z_0)

# Time
t_span = (0, 2e7)
y0 = [R_0, z_0, p_R_0, p_z_0]

solution = solve_ivp(system, t_span, y0, method='DOP853', dense_output=False, 
                     events=poincare_section, rtol=1e-12, atol=1e-14)

# Extract Events
Poincare_R = solution.y_events[0][:, 0]
Poincare_p_R = solution.y_events[0][:, 2]
Poincare_times = solution.t_events[0]

# Compute log return times
return_times = np.diff(Poincare_times)
log_return_times = np.log(return_times)


Poincare_R = Poincare_R[1:]
Poincare_p_R = Poincare_p_R[1:]

# Plot
plt.figure(figsize=(8, 6), dpi=600)
scatter = plt.scatter(Poincare_R, Poincare_p_R, s=0.01, c=log_return_times, cmap='inferno')
cbar = plt.colorbar(scatter)
cbar.set_label("log(T)", fontsize=18)
cbar.ax.tick_params(labelsize=14)

# Energy boundary
R_vals = np.linspace(1.2, 1.8, 500)
p_R_vals = np.linspace(-0.25, 0.25, 500)
R_vals, p_R_vals = np.meshgrid(R_vals, p_R_vals)
boundary = (E + 1 / R_vals)**2 - 1 - (L_z / R_vals - D / R_vals**2)**2 - p_R_vals**2 
plt.contour(R_vals, p_R_vals, boundary, levels=[0], colors='blue')

plt.grid(False)
plt.xlabel("$R$", fontsize=25)
plt.ylabel("$p_R$", fontsize=25)
plt.tick_params(axis='both', labelsize=20)
plt.tight_layout()
plt.show()

end_time = time.time()
print(f"Execution time: {(end_time - start_time)/60} minutes")

