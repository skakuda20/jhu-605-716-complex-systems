import numpy as np
import matplotlib.pyplot as plt

def f(x, alpha):
    return 1 + alpha * np.cos((np.pi / 2) * x**2)

alphas = np.arange(0.001, 1.0001, 0.001)

alpha_values = []
x_values = []

for alpha in alphas:
    x = 1 / alpha

    orbit = []
    for n in range(512):
        x = f(x, alpha)
        orbit.append(x)

    # Last 256 points of orbit
    for x_last in orbit[-256:]:
        alpha_values.append(alpha)
        x_values.append(x_last)

plt.figure(figsize=(10, 6))
plt.plot(alpha_values, x_values, ',k', alpha=0.5)
plt.xlabel('alpha')
plt.ylabel('x_n')
plt.title('Orbit Diagram')
plt.show()