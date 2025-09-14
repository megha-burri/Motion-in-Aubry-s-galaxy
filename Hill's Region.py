import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

# Constants
E = 10
L = -0.2
D = 0.7


R = np.linspace(1e-10, 0.4, 500)  
z = np.linspace(-0.2, 0.2, 500)  
R, Z = np.meshgrid(R, z)

r = np.sqrt(R**2 + Z**2)

# Define Hill's region
Hill = (E + 1/r)**2 - 1 - (L/R - D*R/r**3)**2
boundary = (E +1/r)


plt.figure(figsize=(4, 4), dpi=600)


plt.contourf(R, Z, Hill, levels=[0, np.max(Hill)], colors=['slategrey'], alpha=1.0)


plt.contour(R, Z, Hill, levels=[0], colors='black')


plt.xlabel("R", fontsize = 25)
plt.ylabel("z", fontsize = 25)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.tick_params(axis='both', labelsize=20)

plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


plt.show()
