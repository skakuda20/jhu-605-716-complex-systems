import math
import random
import matplotlib.pyplot as plt

def order_parameter(theta):
    N = len(theta)
    mean_cos = sum(math.cos(t) for t in theta) / N
    mean_sin = sum(math.sin(t) for t in theta) / N

    r = math.sqrt(mean_cos**2 + mean_sin**2)
    psi = math.atan2(mean_sin, mean_cos)
    return r, psi


def kuramoto_rhs(theta, omega, K, alpha):
    r, psi = order_parameter(theta)
    return [omega[i] + K * r * math.sin(psi - theta[i] + alpha)
            for i in range(len(theta))]


def rk4_step(theta, omega, K, alpha, dt):
    N = len(theta)
    k1 = kuramoto_rhs(theta, omega, K, alpha)

    theta2 = [theta[i] + 0.5 * dt * k1[i] for i in range(N)]
    k2 = kuramoto_rhs(theta2, omega, K, alpha)

    theta3 = [theta[i] + 0.5 * dt * k2[i] for i in range(N)]
    k3 = kuramoto_rhs(theta3, omega, K, alpha)

    theta4 = [theta[i] + dt * k3[i] for i in range(N)]
    k4 = kuramoto_rhs(theta4, omega, K, alpha)

    theta_next = [theta[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
                  for i in range(N)]

    return theta_next


def simulate(K, N=1024, T=20.0, dt=0.01, alpha=math.pi/4, seed=None):
    rng = random.Random(seed)
    omega = [rng.uniform(-2 * math.pi, 2 * math.pi) for _ in range(N)]
    theta = [rng.uniform(0, 2 * math.pi) for _ in range(N)]
    steps = int(T / dt)

    for _ in range(steps):
        theta = rk4_step(theta, omega, K, alpha, dt)

    r_final, _ = order_parameter(theta)
    return r_final


def main():
    N = 1024
    T = 20.0
    dt = 0.01
    alpha = math.pi / 4
    threshold = 0.1

    K_values = [round(k * 0.1, 1) for k in range(0, 161)]
    r_values = []
    found = False

    for idx, K in enumerate(K_values):
        r_final = simulate(
            K=K,
            N=N,
            T=T,
            dt=dt,
            alpha=alpha,
            seed=idx
        )

        if r_final > threshold and not found:
            print(f"K_c = {K:.1f}, r_inf ≈ {r_final:.4f}")
            found = True

        r_values.append(r_final)
        print(f"K = {K:.1f}, r_inf ≈ {r_final:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(K_values, r_values, marker="o", markersize=3, linewidth=1)
    plt.xlabel("K")
    plt.ylabel("r")
    plt.title("Kuramoto Model")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()