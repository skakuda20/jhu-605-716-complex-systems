import math
import random

# Compute z and r
sqrt_9936 = math.sqrt(9936)
z = 11/3 + (2/3) * (
    (100 + sqrt_9936) ** (1/3) +
    (100 - sqrt_9936) ** (1/3)
)
r = 1 + math.sqrt(z)

print(f"sqrt(9936) = {sqrt_9936:.10f}")
print(f"z          = {z:.10f}")
print(f"r          = {r:.10f}\n")

# Iterate the logistic map
def logistic(x, r):
    return r * x * (1 - x)

def iterate_sequence(x0, r, n_warmup=1001, n_record=10):
    x = x0
    for _ in range(n_warmup):
        x = logistic(x, r)
    sequence = []
    for i in range(n_record):
        sequence.append(x)
        x = logistic(x, r)
    return sequence

initial_conditions = [0.1, 0.5, 0.9]

for x0 in initial_conditions:
    print(f"        x0={x0:.1f}  ", end="")
print()

for i in range(10):
    print(f"x_{1001+i:04d}", end="")
    for x0 in initial_conditions:
        seq = iterate_sequence(x0, r, n_warmup=1001+i, n_record=1)
        print(f"  {seq[0]:.10f}", end="")
    print()

# Detect periodicity
def detect_period(seq, tol=1e-8):
    for p in range(1, len(seq)):
        if all(abs(seq[i] - seq[i+p]) < tol for i in range(len(seq)-p)):
            return p
    return None

seq = iterate_sequence(0.1, r, n_warmup=2000, n_record=20)
period = detect_period(seq)
print(f"\nDetected period: {period}")
print(f"Cycle values:")
for i in range(period):
    print(f"  point {i+1}: {seq[i]:.10f}")

# Plot bifurcation diagram
import matplotlib.pyplot as plt
import numpy as np

r_values = np.linspace(2.8, 4.0, 1000)
warmup, n_plot = 500, 50

fig, ax = plt.subplots(figsize=(10, 5))
for rv in r_values:
    x = 0.4
    for _ in range(warmup):
        x = rv * x * (1 - x)
    xs = []
    for _ in range(n_plot):
        xs.append(x)
        x = rv * x * (1 - x)
    ax.plot([rv]*n_plot, xs, ',k', alpha=0.15, markersize=0.5)

ax.axvline(r, color='red', linewidth=1.2, linestyle='--', label=f'r = {r:.6f}')
ax.set_xlabel('r')
ax.set_ylabel('x')
ax.set_title('Logistic Map — Orbit Diagram')
ax.legend()
plt.tight_layout()
plt.savefig('bifurcation.png', dpi=150)
plt.show()