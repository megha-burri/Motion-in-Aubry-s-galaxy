import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Constants
D = 3
delta = 1e-7

# v ranges
v_pos = np.linspace(delta, 1 - delta, 1000000)
v_neg = np.linspace(-1 + delta, -delta, 1000000)
v_all = np.linspace(delta, 1 - delta, 1000000)

# Functions
def gamma(v):
    return 1 / np.sqrt(1 - v**2)

def sqrt_expr(v):
    return np.sqrt(1 + 4 * D * gamma(v) * v**3)

def Eplus(v):
    return gamma(v) + (1 / (2 * D * v)) * (1 - sqrt_expr(v)) - 1

def Lzplus(v):
    return (1 / v) * sqrt_expr(v)

def Eminus(v):
    return gamma(v) + (1 / (2 * D * v)) * (1 + sqrt_expr(v)) - 1

def Lzminus(v):
    return -(1 / v) * sqrt_expr(v)

def E3(v):
    return (1 - 1.5 * v**2) / np.sqrt(1 - v**2) - 1

def Lz3(v):
    return -2 * np.sqrt(1 - v**2) / (27 * D * v**4)

def constraint(v):
    return v**6 >= 4 * (1 - v**2) / (81 * D**2)


v3 = v_all[constraint(v_all)]

# Compute values
Eplus_pos, Lzplus_pos = Eplus(v_pos), Lzplus(v_pos)
Eminus_pos, Lzminus_pos = Eminus(v_pos), Lzminus(v_pos)
Eplus_neg, Lzplus_neg = Eplus(v_neg), Lzplus(v_neg)
Eminus_neg, Lzminus_neg = Eminus(v_neg), Lzminus(v_neg)
E3_vals, Lz3_vals = E3(v3), Lz3(v3)

# Figure size
fig_width = 7.0     
fig_height = 8.0    
dpi = 600           

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5
})

fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

# Plot curves
col = '#FF4D00'
lw = 1.0

ax.plot(Eplus_pos, Lzplus_pos, color=col, linewidth=lw)
ax.plot(Eminus_pos, Lzminus_pos, color=col, linewidth=lw)
ax.plot(Eplus_neg, Lzplus_neg, color=col, linewidth=lw)
ax.plot(Eminus_neg, Lzminus_neg, color=col, linewidth=lw)
ax.plot(E3_vals, Lz3_vals, color=col, linewidth=lw)

# Axis formatting
ax.set_xlim([-2, 2])
ax.set_ylim([-2.5, 4])
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
xticks = np.arange(-2, 3, 1)
yticks = np.arange(-2, 6, 1)

def format_ticks(x, pos):
    if np.isclose(x, 0):
        return ""
    return f"{int(x)}" if x == int(x) else f"{x:.1f}"

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
ax.tick_params(labelsize=10)

# Axis labels
ax.annotate(r'$E-1$', xy=(2, 0.2), fontsize=12, ha='right')
ax.annotate(r'$L_z$', xy=(-0.2, 4.9), fontsize=12, va='top')
ax.text(-0.1, -0.2, '0', fontsize=10)


# Selected values in parameter space
points = np.array([
    (0.3 - 1, 1.0),
    (0.85 - 1, 3.4),
    (1.1 - 1, 4.3),
    (0.95 - 1, 3.4),
    (2 - 1, 2),
    (0.9 - 1, -0.7),
    (0.7 - 1, -0.1),
    (2.9 - 1, -0.5)
])
ax.scatter(points[:, 0], points[:, 1], color='blue', s=2.5, zorder=5)

plt.tight_layout()
plt.show()

