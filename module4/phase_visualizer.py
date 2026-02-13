import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

# -----------------------------
# Define the system
# -----------------------------
def system(x, y):
    dx = x * (x - y)
    dy = y * (2*x - y)
    return dx, dy

# -----------------------------
# Jacobian matrix
# -----------------------------
def jacobian(x, y):
    return np.array([
        [2*x - y, -x],
        [2*y, 2*x - 2*y]
    ])

# -----------------------------
# Fixed Point Calculation
# -----------------------------
fixed_points = [(0, 0)]

print("Fixed Point Analysis:\n")

for fp in fixed_points:
    J = jacobian(fp[0], fp[1])
    eigenvalues, _ = eig(J)
    
    print(f"Fixed Point: {fp}")
    print("Jacobian:\n", J)
    print("Eigenvalues:", eigenvalues)
    
    if np.allclose(eigenvalues, 0):
        print("Type: Nonhyperbolic (linearization inconclusive)\n")
    elif np.any(np.real(eigenvalues) > 0) and np.any(np.real(eigenvalues) < 0):
        print("Type: Saddle\n")
    elif np.all(np.real(eigenvalues) < 0):
        print("Type: Stable\n")
    elif np.all(np.real(eigenvalues) > 0):
        print("Type: Unstable\n")
    else:
        print("Type: Indeterminate\n")

# -----------------------------
# Plot Nullclines
# -----------------------------
x = np.linspace(-3, 3, 400)

plt.figure()
plt.plot(x, x)       # y = x
plt.plot(x, 2*x)     # y = 2x
plt.axhline(0)       # y = 0
plt.axvline(0)       # x = 0
plt.title("Nullclines")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# -----------------------------
# Plot Vector Field
# -----------------------------
x_vals = np.linspace(-3, 3, 20)
y_vals = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x_vals, y_vals)

U, V = system(X, Y)

plt.figure()
plt.quiver(X, Y, U, V)
plt.title("Vector Field")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# -----------------------------
# Plot Phase Portrait
# -----------------------------
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

U, V = system(X, Y)

plt.figure()
plt.streamplot(X, Y, U, V, density=1)
plt.scatter(0, 0)
plt.title("Phase Portrait with Fixed Point")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
