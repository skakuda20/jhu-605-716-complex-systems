import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
T = 1000            # time steps
transient = 200     # transient time to discard
y0 = [1, 2, 3]      # initial conditions
alphas = [0.30, 0.22, 0.20, 0.10]


def system(t, state, alpha):
    x, y, z = state
    dxdt = -alpha * x + np.sin(y)
    dydt = -alpha * y + np.sin(z)
    dzdt = -alpha * z + np.sin(x)
    return [dxdt, dydt, dzdt]

def simulate(alpha, T=300, dt=0.01, transient=100):
    t_eval = np.arange(0, T, dt)

    sol = solve_ivp(
        system,
        t_span=(0, T),
        y0=y0,
        args=(alpha,),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9
    )

    t = sol.t
    x, y, z = sol.y
    final_state = (x[-1], y[-1], z[-1])
    print(f"Final state for alpha={alpha}: {final_state}")
    
    mask = t > transient
    return t[mask], x[mask], y[mask], z[mask]

def plot_results(alpha, T=1000, transient=100):
    t, x, y, z = simulate(alpha, T=T, transient=transient)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.ylabel("x(t)")
    plt.title(f"Time Series for alpha = {alpha} | Initial Conditions: {y0}")

    plt.subplot(3, 1, 2)
    plt.plot(t, y)
    plt.ylabel("y(t)")

    plt.subplot(3, 1, 3)
    plt.plot(x, z)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Phase Plane: x vs z")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for alpha in alphas:
        plot_results(alpha, T=T, transient=transient)