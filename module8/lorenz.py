import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters (edit these)
# ----------------------------
sigma = 10.0
r = 400.0
b = 8.0 / 3.0

dt = 0.001
T = 50
N = int(T / dt)

# Initial condition
x0, y0, z0 = 1.0, 1.0, 1.0

# ----------------------------
# Lorenz system definition
# ----------------------------
def lorenz(state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return np.array([dx, dy, dz])

# ----------------------------
# RK4 Integrator
# ----------------------------
def rk4_step(state, dt):
    k1 = lorenz(state)
    k2 = lorenz(state + 0.5 * dt * k1)
    k3 = lorenz(state + 0.5 * dt * k2)
    k4 = lorenz(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ----------------------------
# Time integration
# ----------------------------
t = np.linspace(0, T, N)
trajectory = np.zeros((N, 3))

state = np.array([x0, y0, z0])

for i in range(N):
    trajectory[i] = state
    state = rk4_step(state, dt)

x = trajectory[:, 0]
y = trajectory[:, 1]
z = trajectory[:, 2]

# ----------------------------
# Plotting
# ----------------------------
plt.figure()
plt.plot(t, x)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("x(t)")
plt.grid()

plt.figure()
plt.plot(t, y)
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("y(t)")
plt.grid()

plt.figure()
plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("z")
plt.title("Phase Plot: x vs z")
plt.grid()

plt.show()