import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

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

