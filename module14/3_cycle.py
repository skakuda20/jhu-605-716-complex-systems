import numpy as np

def f(x, alpha):
    return 1 + alpha * np.cos((np.pi / 2) * x**2)

alpha = 0.82
x = 1 / alpha

for n in range(5000):
    x = f(x, alpha)

cycle = []
for n in range(3):
    x = f(x, alpha)
    cycle.append(x)

print(cycle)