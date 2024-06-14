import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from fractions import Fraction

# Define symbols
theta, F = sp.symbols('theta F')

# Given parameters
M = 1
m = 0.1
l = 1
g = 9.81

# Equation for the cart-pole system
ddot_theta = (g * sp.sin(theta) * (M + m) - (F + m * l * sp.diff(theta)**2 * sp.sin(theta)) * sp.cos(theta)) / \
             (l * ((4/3) * (M + m) - m * sp.cos(theta)**2))

# Partial derivative of ddot_theta with respect to F
ddot_theta_F = sp.diff(ddot_theta, F)

# Substituting given values
ddot_theta_F = ddot_theta_F.subs({M: 1, m: 0.1, l: 1, g: 9.81})

# Solve for roots
roots = sp.solve(ddot_theta_F, F)

# Plot
theta_vals = np.linspace(-3*np.pi, 3*np.pi, 400)
F_vals = [ddot_theta_F.subs({theta: theta_val}) for theta_val in theta_vals]

plt.figure(figsize=(8, 6))
plt.plot(theta_vals, F_vals, color='blue')
plt.xlabel(r'$\theta$', fontsize=15, fontweight='bold', rotation=0, ha='center')
plt.ylabel(r'$\frac{\partial \ddot{\theta}}{\partial F}$', fontsize=15, fontweight='bold', rotation=1, ha='center', va='center')
plt.title('Partial Derivative of $\ddot{\\theta}$ with respect to $F$')
plt.axhline(0, color='black', linewidth=0.9)
plt.axvline(0, color='black', linewidth=0.9)

# Find where the curve crosses the axes
crossings = [theta_vals[i] for i in range(len(F_vals)-1) if F_vals[i] * F_vals[i+1] < 0]

# Mark where the curve crosses x-axis and annotate the null points
for crossing in crossings:
    plt.scatter(crossing, 0, color='red', zorder=5)
    plt.annotate(f"~{crossing/np.pi:.2f}π", (crossing, 0), xytext=(5, -15), textcoords='offset points', ha='center', fontsize=8)

plt.grid(True, linestyle='--', alpha=0.7)
plt.grid(True)
plt.show()

