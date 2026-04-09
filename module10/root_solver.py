import numpy as np
from scipy.optimize import fsolve

# Parameters
alpha = 0.25

# Define map
def f(x):
    return 0.5 + alpha * np.sin(2 * np.pi * x)

# Second iterate
def f2(x):
    return f(f(x))

# Equation for 2-cycles
def g(x):
    return f2(x) - x

# Derivative of f
def df(x):
    return 2 * np.pi * alpha * np.cos(2 * np.pi * x)

# Find roots 
guesses = np.linspace(0, 1, 20)
roots = []

for guess in guesses:
    root = fsolve(g, guess)[0] % 1  # keep in [0,1)
    # avoid duplicates
    if not any(abs(root - r) < 1e-5 for r in roots):
        roots.append(root)
roots = sorted(roots)

print("All solutions to f(f(x)) = x:")
for r in roots:
    print(f"{r:.6f}")

fixed_points = []
two_cycle = []

for r in roots:
    if abs(f(r) - r) < 1e-5:
        fixed_points.append(r)
    else:
        two_cycle.append(r)

print("\nFixed points:")
for r in fixed_points:
    print(f"{r:.6f}")

print("\n2-cycle points:")
for r in two_cycle:
    print(f"{r:.6f}")

# --- compute stability of 2-cycle ---
print("\nStability of 2-cycle:")
for x in two_cycle:
    multiplier = df(x) * df(f(x))
    print(f"x = {x:.6f}, (f^2)' = {multiplier:.6f}, |.| = {abs(multiplier):.6f}")