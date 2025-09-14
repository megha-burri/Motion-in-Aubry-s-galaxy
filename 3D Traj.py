import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Constants
M = 1.0  
D = 10

def system(t, state):
    x, y, z, px, py, pz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    gamma = np.sqrt(1 + px**2 + py**2 + pz**2)

    dxdt = px / gamma
    dydt = py / gamma
    dzdt = pz / gamma

    dpxdt = ((-M * x / r**3)
             + (D / r**5) * (3 * z**2 - r**2) * (py / gamma)
             - 3 * D * y * z * pz / (gamma * r**5))
    dpydt = (-M * y / r**3
             - (D / r**5) * (3 * z**2 - r**2) * (px / gamma)
             + 3 * D * x * z * pz / (gamma * r**5))
    dpzdt = (-M * z / r**3
             + 3 * (D * z / (gamma * r**5)) * (y * px - x * py))
    
    return [dxdt, dydt, dzdt, dpxdt, dpydt, dpzdt]

# Initial conditions
x0, y0, z0 = 0.0, 1.0, 0.0
px0, py0, pz0 = 0.5, 0.0, 0.5
initial_state = [x0, y0, z0, px0, py0, pz0]

#Time 
t_span = (0, 50)

#Solve 
solution = solve_ivp(system, t_span, initial_state, method='LSODA', rtol=1e-10, atol=1e-12)
x, y, z = solution.y[0], solution.y[1], solution.y[2]
    
# Plot 3D trajectory
fig = plt.figure(figsize=(10, 10), dpi = 800)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color = 'blue')
ax.set_xlabel("$x$", fontsize = 25, labelpad = 15)
ax.set_ylabel("$y$", fontsize = 25, labelpad = 15)
ax.set_zlabel("$z$", fontsize = 25, labelpad = 15)
plt.tick_params(axis='both', labelsize=20)

ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

angle_elev = 30
angle_azim = -60
ax.view_init(elev=angle_elev, azim=angle_azim)
ax.set_box_aspect(None, zoom=0.85)
plt.show()