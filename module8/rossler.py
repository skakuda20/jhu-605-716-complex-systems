import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# parameters
a = 0.2
b = 0.2
c = 6.3   # change this for different parts

# rossler system
def rossler(state):
    x, y, z = state
    dx = -(y + z)
    dy = x + a*y
    dz = b + x*z - c*z
    return np.array([dx, dy, dz])

# RK4 integrator
def rk4_step(f, state, dt):
    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# simulation settings
dt = 0.01
T = 100
steps = int(T/dt)

state = np.array([1.0,1.0,1.0])

traj = np.zeros((steps,3))

for i in range(steps):
    traj[i] = state
    state = rk4_step(rossler,state,dt)

# Identify the attractor: discard transient behavior (first 10000 steps ~100 time units)
transient_cutoff = 0
traj_attractor = traj[transient_cutoff:]

x = traj_attractor[:,0]
y = traj_attractor[:,1]
z = traj_attractor[:,2]
t = np.linspace(0, T, steps)[transient_cutoff:]

# plot x(t)
plt.figure()
plt.plot(t,x)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("x(t)")
plt.show()

# phase plane
plt.figure()
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"phase plane c={c}")
plt.show()

# 3D attractor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title(f"Rössler Attractor c={c}")
plt.show()

# TODO: refine code
# TODO: change title to include parameter values