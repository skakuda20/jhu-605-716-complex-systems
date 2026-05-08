import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def rpslk_system(t, y):
    R, P, S, L, K = y

    dR = R * (S + L - P - K)
    dP = P * (R + K - S - L)
    dS = S * (P + L - R - K)
    dL = L * (P + K - R - S)
    dK = K * (R + S - P - L)

    return [dR, dP, dS, dL, dK]


def simulate_and_plot(y0):
    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 2000)

    sol = solve_ivp(
        rpslk_system,
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-12
    )

    R, P, S, L, K = sol.y

    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, R, label="Rock")
    plt.plot(sol.t, P, label="Paper")
    plt.plot(sol.t, S, label="Scissors")
    plt.plot(sol.t, L, label="Lizard")
    plt.plot(sol.t, K, label="Spock")

    plt.xlabel("Time")
    plt.ylabel("Population fraction")
    plt.title(f"RPSLK | Initial condition: {y0}")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    y0_d = [91/100, 1/25, 3/100, 1/50, 0]
    simulate_and_plot(y0_d)

    y0_e = [1/2, 49/100, 0, 1/100, 0]
    simulate_and_plot(y0_e)