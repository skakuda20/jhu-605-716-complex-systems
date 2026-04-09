import numpy as np
import matplotlib.pyplot as plt

def iterate_map(z0, n):
    z = z0
    zs = []

    for _ in range(n):
        z = np.pi + 0.5 * z * np.exp(1j * abs(z)**2)
        zs.append(z)

    return np.array(zs)


# Find attractor
z0 = np.random.rand() + 1j * np.random.rand()
zs = iterate_map(z0, 5000)

# Take a point after transient
z_attractor = zs[-1]


# Generate final plot
N = 2**15
zs = iterate_map(z_attractor, N)

# Extract real and imaginary parts
x = zs.real
y = zs.imag

# Plot
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=0.1)

plt.xlim(0, 2*np.pi)
plt.ylim(-np.pi, np.pi)

plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title("Attractor of Complex Iterated Map")

plt.show()
